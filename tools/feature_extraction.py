# Copyright 2018-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from caffe2.python import workspace, cnn, data_parallel_model, core
from caffe2.proto import caffe2_pb2


from models import model_builder
from utils import model_helper
from utils import model_loader
from utils import metric
from utils import reader_utils


import numpy as np
import logging
import argparse
import os.path
import pickle
import h5py

logging.basicConfig()
log = logging.getLogger("feature_extractor")
log.setLevel(logging.INFO)

def feature_extractor(load_model_path=None, test_data=None, gpu_list=None, num_gpus=0, batch_size=4, clip_per_video=1, decode_type=2
                        , clip_length_rgb=4, sampling_rate_rgb=1, scale_h=128, scale_w=171
                        , crop_size=112, video_res_type=0, num_decode_threads=4, multi_label=0
                        , num_labels=101, input_type=0, clip_length_of=8, sampling_rate_of=2
                        , frame_gap_of=2, do_flow_aggregation=0, flow_data_type=0
                        , get_video_id=1, get_start_frame=0, use_local_file=1, crop_per_clip=1
                        , db_type='pickle' , model_name='r2plus1d', model_depth=18
                        , num_channels=3, output_path=None
                        , use_cudnn=1, layers='final_avg', num_iterations=1, channel_multiplier=1.0
                        , bottleneck_multiplier=1.0, use_pool1=0, use_convolutional_pred=0, use_dropout=0, **kwargs):
    """
    :param gpu_list: list of gpu ids to use
    :param batch_size: batch size
    :param clip_per_video: When clips_per_video > 1, sample this many clips uniformly in time
    :param decode_type: 0: random, 1: uniform sampling, 2: use starting frame
    :param clip_length_rgb: Length of input clips
    :param sampling_rate_rgb: Frame sampling rate
    :param scale_h: Scale image height to
    :param scale_w: Scale image width to
    :param crop_size: Input image size (to crop to)
    :param video_res_type: Video frame scaling option, 0: scaled by height x width; 1: scaled by short edge
    :param num_decode_threads: number of decoding threads
    :param multi_label: Multiple label csv file input
    :param num_labels: Number of labels
    :param input_type: 0=rgb, 1=optical flow
    :param clip_length_of: Frames of optical flow data
    :param sampling_rate_of: Sampling rate for optial flows
    :param frame_gap_of: Frame gap of optical flows
    :param do_flow_aggregation: whether to aggregate optical flow across multiple frames
    :param flow_data_type: 0=Flow2C, 1=Flow3C, 2=FlowWithGray, 3=FlowWithRGB
    :param get_video_id: Output video id
    :param get_start_frame: Output clip start frame
    :param use_local_file: use local file
    :param crop_per_clip: number of spatial crops per clip
    :param db_type: Db type of the testing model
    :param model_name: Model name
    :param model_depth: Model depth
    :param num_channels: Number of channels
    :param load_model_path: Load saved model for testing
    :param test_data: Path to output pickle; defaults to layers.pickle next to <test_data>
    :param output_path: Path to output pickle; defaults to layers.pickle next to <test_data>
    :param use_cudnn: Use CuDNN
    :param layers: Comma-separated list of blob names to fetch
    :param num_iterations: Run only this many iterations
    :param channel_multiplier: Channel multiplier
    :param bottleneck_multiplier: Bottleneck multiplier
    :param use_pool1: use pool1 layer
    :param use_convolutional_pred: using convolutional predictions
    :param use_dropout: Use dropout at the prediction layer
    """
    if load_model_path is None or test_data is None:
        raise Exception('Model path AND test data need to be specified')

    # Initialize Caffe2
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])

    if gpu_list is None:
        if num_gpus == 0:
            raise Exception('Must specify GPUs')
        else:
            gpus = [i for i in range(num_gpus)]
    else:
        gpus = gpu_list
        num_gpus = len(gpus)

    my_arg_scope = {
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustive_search': True
    }

    model = cnn.CNNModelHelper(
        name="Extract features",
        **my_arg_scope
    )

    video_input_args = dict(
        batch_size=batch_size,
        clip_per_video=clip_per_video,
        decode_type=decode_type,
        length_rgb=clip_length_rgb,
        sampling_rate_rgb=sampling_rate_rgb,
        scale_h=scale_h,
        scale_w=scale_w,
        crop_size=crop_size,
        video_res_type=video_res_type,
        short_edge=min(scale_h, scale_w),
        num_decode_threads=num_decode_threads,
        do_multi_label=multi_label,
        num_of_class=num_labels,
        random_mirror=False,
        random_crop=False,
        input_type=input_type,
        length_of=clip_length_of,
        sampling_rate_of=sampling_rate_of,
        frame_gap_of=frame_gap_of,
        do_flow_aggregation=do_flow_aggregation,
        flow_data_type=flow_data_type,
        get_rgb=input_type == 0,
        get_optical_flow=input_type == 1,
        get_video_id=get_video_id,
        get_start_frame=get_start_frame,
        use_local_file=use_local_file,
        crop_per_clip=crop_per_clip,
    )

    reader_args = dict(
        name="extract_features" + '_reader',
        input_data=test_data,
    )

    reader, num_examples = reader_utils.create_data_reader(
        model,
        **reader_args
    )

    def input_fn(model):
        model_helper.AddVideoInput(
            model,
            reader,
            **video_input_args)

    def create_model_ops(model, loss_scale):
        return model_builder.build_model(
            model=model,
            model_name=model_name,
            model_depth=model_depth,
            num_labels=num_labels,
            batch_size=batch_size,
            num_channels=num_channels,
            crop_size=crop_size,
            clip_length=(
                clip_length_of if input_type == 1
                else clip_length_rgb
            ),
            loss_scale=loss_scale,
            is_test=1,
            multi_label=multi_label,
            channel_multiplier=channel_multiplier,
            bottleneck_multiplier=bottleneck_multiplier,
            use_dropout=use_dropout,
            use_convolutional_pred=use_convolutional_pred,
            use_pool1=use_pool1,
        )

    if num_gpus > 0:
        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=input_fn,
            forward_pass_builder_fun=create_model_ops,
            param_update_builder_fun=None,   # 'None' since we aren't training
            devices=gpus,
            optimize_gradient_memory=True,
        )
    else:
        model._device_type = caffe2_pb2.CPU
        model._devices = [0]
        device_opt = core.DeviceOption(model._device_type, 0)
        with core.DeviceScope(device_opt):
            # Because our loaded models are named with "gpu_x", keep the naming for now.
            # TODO: Save model using `data_parallel_model.ExtractPredictorNet`
            # to extract the model for "gpu_0". It also renames
            # the input and output blobs by stripping the "gpu_x/" prefix
            with core.NameScope("{}_{}".format("gpu", 0)):
                input_fn(model)
                create_model_ops(model, 1.0)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    if db_type == 'pickle':
        model_loader.LoadModelFromPickleFile(model, load_model_path)
    elif db_type == 'minidb':
        if num_gpus > 0:
            model_helper.LoadModel(load_model_path, db_type)
        else:
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
                model_helper.LoadModel(load_model_path, db_type)
    else:
        log.warning("Unsupported db_type: {}".format(db_type))

    data_parallel_model.FinalizeAfterCheckpoint(model)

    def fetchActivations(model, outputs, num_iterations):

        all_activations = {}
        for counter in range(num_iterations):
            workspace.RunNet(model.net.Proto().name)

            num_devices = 1  # default for cpu
            if num_gpus > 0:
                num_devices = num_gpus
            for g in range(num_devices):
                for output_name in outputs:
                    blob_name = 'gpu_{}/'.format(g) + output_name
                    activations = workspace.FetchBlob(blob_name)
                    if output_name not in all_activations:
                        all_activations[output_name] = []
                    all_activations[output_name].append(activations)

        # each key holds a list of activations obtained from each minibatch.
        # we now concatenate these lists to get the final arrays.
        # concatenating during the loop requires a realloc and can get slow.
        for key in all_activations:
            all_activations[key] = np.concatenate(all_activations[key])

        return all_activations

    # outputs = [name.strip() for name in layers.split(',')]
    if not isinstance(layers, list):
        layers = [layers]
    assert len(layers) > 0

    if num_iterations > 0:
        num_iterations = num_iterations
    else:
        if num_gpus > 0:
            examples_per_iteration = batch_size * num_gpus
        else:
            examples_per_iteration = args.batch_size
        num_iterations = int(num_examples / examples_per_iteration)

    activations = fetchActivations(model, layers, num_iterations)

    # saving extracted layers
    for index in range(len(layers)):
        log.info(
            "Read '{}' with shape {}".format(
                layers[index],
                activations[layers[index]].shape
            )
        )

    if output_path:
        log.info('Writing to {}'.format(output_path))
        if args.save_h5:
            with h5py.File(output_path, 'w') as handle:
                for name, activation in activations.items():
                    handle.create_dataset(name, data=activation)
        else:
            with open(output_path, 'wb') as handle:
                pickle.dump(activations, handle)
    else:
        return activations
