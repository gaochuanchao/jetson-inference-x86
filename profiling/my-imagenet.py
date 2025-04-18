#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import time
import csv
import sys
import argparse
import subprocess

from jetson_inference import imageNet
from jetson_utils import loadImage
from jetson_utils import cudaDeviceSynchronize


if __name__ == '__main__':
    # load the recognition network
    # note: to hard-code the paths to load a model, the following API can be used:
    #
    # net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
    #                 input_blob="input_0", output_blob="output_0")
    # classification networks:
    # alexnet, googlenet, googlenet-12, resnet-18,	resnet-50, resnet-101, resnet-152, vgg-16, vgg-19, inception-v4

    # parse the command line
    parser = argparse.ArgumentParser(description="Profiling imagenet", 
                                    formatter_class=argparse.RawTextHelpFormatter, 
                                    epilog=imageNet.Usage())

    # check network, if test run, power mode
    parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
    parser.add_argument("--profile", type=str, default="False", nargs='?', help="whether this is a test run")
    parser.add_argument("--gpu", type=str, default="4090", nargs='?', help="the GPU type")

    try:
        args = parser.parse_known_args()[0]
        print("network: ", args.network, "; profile: ", args.profile)
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    precision = "FP16"  # FP32, FP16
    model_type = "imagenet"
    # load images (into shared CPU/GPU memory)
    with open("img_name.txt", 'r') as f_name:
        images = f_name.read().split()
    
    folder_path = "../images/"

    if args.network == "all":
        network_names = ["alexnet", "googlenet", "googlenet-12", "resnet-18", "resnet-50", "resnet-101", "resnet-152", "vgg-16", "vgg-19", "inception-v4"]
    else:
        network_names = [args.network]
    
    for network_name in network_names:
        try:
            print("==========================")
            print("network: ", network_name)
            print("==========================")
            time.sleep(5)   # wait for 2s
            # network_name = args.network
            net = imageNet(network_name)
            # warm up run
            for image in images[:20]:
                try:
                    img = loadImage(folder_path + image)   # load image
                    class_idx, confidence = net.Classify(img)
                    cudaDeviceSynchronize()  # <--- Ensures GPU work is complete
                    # time.sleep(0.003)
                    exe_time = net.GetNetworkTime()
                    overall_time = net.GetCPUTotalTime()
                    class_desc = net.GetClassDesc(class_idx)
                    print("class_desc:", class_desc, "; confidence:", confidence)
                    print("exe_time:", exe_time, "; overall_time:", overall_time)
                except:
                    pass

            # net.PrintProfilerTimes()  # print the profiler times

            # check if this is a test run
            profile_flag = False
            if (args.profile == "True"):
                profile_flag = True
            
            REPEAT_COUNT = 50 # 50  
            all_results = []

            if profile_flag:
                for files in images:      # start profiling
                    result = {'image': files, 'start': 0, "end": 0, 'time': [], 'cpu_time': []}
                    img_path = folder_path + files
                    # classify the image and get the prediction
                    print("start inferencing", files, "...")
                    result["start"] = round(time.time(), 3)
                    cnt = 0
                    while cnt < REPEAT_COUNT:
                        try:
                            img = loadImage(img_path)   # load image
                            class_idx, confidence = net.Classify(img)
                            cudaDeviceSynchronize()  # <--- Ensures GPU work is complete
                            # time.sleep(0.003)
                            exe_time = net.GetNetworkTime()
                            overall_time = net.GetCPUTotalTime()
                            # class_desc = net.GetClassDesc(class_idx)
                            result["time"].append(round(exe_time, 2))
                            result["cpu_time"].append(round(overall_time, 2))
                            cnt += 1
                        except Exception:
                            pass
                    result["end"] = round(time.time(), 3)
                    all_results.append(result)
                    print("end inferencing", files, "...")
                
                print("start saving results ...")
                with open("{}/runtime/{}_runtime_{}_{}_{}.csv".format(precision, args.gpu, model_type, network_name, precision), "w", newline='') as csvfile:
                    csvkeys = ['image', "start", "end"]
                    for i in range(REPEAT_COUNT):
                        csvkeys.append("run_"+str(i))
                    for i in range(REPEAT_COUNT):
                        csvkeys.append("cpu_time_"+str(i))
                    writer = csv.DictWriter(csvfile, fieldnames=csvkeys)
                    writer.writeheader()    # Write header
                    for result in all_results:
                        output = dict()
                        output['image'] = result['image']
                        output["start"] = result["start"]
                        output["end"] = result["end"]
                        for i in range(REPEAT_COUNT):
                            key_name = "run_"+str(i)
                            output[key_name] = result["time"][i]
                        for i in range(REPEAT_COUNT):
                            key_name = "cpu_time_"+str(i)
                            output[key_name] = result["cpu_time"][i]
                        writer.writerow(output)

        finally:
            if (profile_flag):
                time.sleep(2)   # wait for 2s
                ave_time = []
                ave_cpu_time = []
                exe_time = []
                power = dict()
                with open("{}/runtime/{}_runtime_{}_{}_{}.csv".format(precision, args.gpu, model_type, network_name, precision), 'r') as f1:
                    f1.readline()
                    while True:
                        line = f1.readline()
                        if not line:
                            break
                        line = line.strip()
                        data = line.split(',')
                        exe_time.append((float(data[1]), float(data[2])))
                        runtime = [float(data[i]) for i in range(3, 3+REPEAT_COUNT)]
                        ave_time.append(round(sum(runtime)/len(runtime), 2))
                        cpu_runtime = [float(data[i]) for i in range(3+REPEAT_COUNT, 3+2*REPEAT_COUNT)]
                        ave_cpu_time.append(round(sum(cpu_runtime)/len(cpu_runtime), 2))

                with open("{}/results/{}_average_runtime_{}_{}_{}.csv".format(precision, args.gpu, model_type, network_name, precision), 'w') as f_result:
                    f_result.write("avg_time, avg_cpu_time\n")
                    for i in range(len(ave_time)):
                        f_result.write(f"{ave_time[i]}, {ave_cpu_time[i]}\n")

                print("saved to ", "{}/results/{}_average_runtime_{}_{}_{}.csv".format(precision, args.gpu, model_type, network_name, precision))
