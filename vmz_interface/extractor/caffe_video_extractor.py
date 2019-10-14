from vmz_interface.tools.feature_extraction import feature_extractor

import argparse

class DefaultVideoArgs:
	clip_per_video=1
	decode_type=2    
	clip_length_rgb=4
	sampling_rate_rgb=1
	scale_h=128
	scale_w=171    
	crop_size=112
	video_res_type=0
	num_decode_threads=4
	multi_label=0    
	num_labels=101
	input_type=0
	clip_length_of=8
	sampling_rate_of=2    
	frame_gap_of=2
	do_flow_aggregation=0
	flow_data_type=0    
	get_video_id=1
	get_start_frame=1
	use_local_file=1
	crop_per_clip=1    
	db_type='pickle'   
	num_channels=3
	output_path=None
	use_cudnn=1
	num_iterations=1
	channel_multiplier=1.0    
	bottleneck_multiplier=1.0
	use_pool1=0
	use_convolutional_pred=0
	use_dropout=0

class CaffeVideoWrapper:

	def __init__(self, model_name, model_depth, load_model_path, batch_size
				, data_inputs, **kwargs):
		"""
		kwargs:
		clip_per_video=DefaultVideoArgs.clip_per_video, decode_type=DefaultVideoArgs.decode_type
		, clip_length_rgb=clip_length_rgb, sampling_rate_rgb=DefaultVideoArgs.sampling_rate_rgb
		, scale_h=DefaultVideoArgs.scale_h, scale_w=DefaultVideoArgs.scale_w, crop_size=DefaultVideoArgs.crop_size
		, video_res_type=DefaultVideoArgs.video_res_type, num_decode_threads=DefaultVideoArgs.num_decode_threads
		, multi_label=DefaultVideoArgs.multi_label, num_labels=DefaultVideoArgs.num_labels
		, input_type=DefaultVideoArgs.input_type, clip_length_of=DefaultVideoArgs.clip_length_of
		, sampling_rate_of=DefaultVideoArgs.sampling_rate_of, frame_gap_of=DefaultVideoArgs.frame_gap_of
		, do_flow_aggregation=DefaultVideoArgs.do_flow_aggregation, flow_data_type=DefaultVideoArgs.flow_data_type
		, get_video_id=DefaultVideoArgs.get_video_id, get_start_frame=DefaultVideoArgs.get_start_frame
		, use_local_file=DefaultVideoArgs.use_local_file, crop_per_clip=DefaultVideoArgs.crop_per_clip
		, db_type=DefaultVideoArgs.db_type, num_channels=DefaultVideoArgs.num_channels
		, output_path=DefaultVideoArgs.output_path, use_cudnn=DefaultVideoArgs.use_cudnn
		, num_iterations=DefaultVideoArgs.num_iterations, channel_multiplier=DefaultVideoArgs.channel_multiplier
		, bottleneck_multiplier=DefaultVideoArgs.bottleneck_multiplier, use_pool1=DefaultVideoArgs.use_pool1
		, use_convolutional_pred=DefaultVideoArgs.use_convolutional_pred, use_dropout=DefaultVideoArgs.use_dropout
		"""

		self._data_inputs = data_inputs
		self._gpus = [i for i in range(self._data_inputs.GPU_CNT)]
		self.load_model_path = load_model_path
		self.model_name = model_name
		self.model_depth = model_depth
		self._mdl_params = kwargs

	def __call__(self, layers, stimulus_paths):
		self._data_inputs.make_from_paths(stimulus_paths)
		return self._get_activations(layers)


	def _get_activations(self, layer_names):

		a = []

		if self._data_inputs.gpu_batch_combo is None:
			for i in self._data_inputs.video_lmdb_paths:
				# vmz extract_features:
				a.append(feature_extractor(self.model_name, self.model_depth, gpu_list=self._gpus
					, load_model_path=self.load_model_path, test_data=i
					, batch_size=self._data_inputs.BATCH_SIZE, layers=layer_names
					, **self._mdl_params))
		else:
			for i in self._data_inputs.video_lmdb_paths[:-1]:
				# vmz extract_features:
				a.append(feature_extractor(self.model_name, self.model_depth, num_gpu=self._gpus
					, load_model_path=self.load_model_path, test_data=i
					, batch_size=self._data_inputs.BATCH_SIZE, layers=layer_names
					, **self._mdl_params))

			a.append(feature_extractor(self.model_name, self.model_depth
				, num_gpu=self._data_inputs.gpu_batch_combo[0], load_model_path=self.load_model_path
				, test_data=i, batch_size=self._data_inputs.gpu_batch_combo[1], layers=layer_names
				, **self._mdl_params))

		return a
