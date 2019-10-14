import os
import csv
import numpy as np
import pandas as pd
import pickle as pk
from glob import glob
from math import ceil

from vmz_interface.data.db_video_create import create_video_db

class VideoDBBuilder:

	def __init__(self, stimulus_id, lmdb_path, temporal_depth, fpv=75, video_strt_offset=15, clips_overlap=0
					 , batch_size=4, gpu_count=2, max_num_records=6e4, min_records_factor=1, allow_mkdir=False
					 , *args, **kwargs):
		
		if not os.path.isdir(lmdb_path):
			if allow_mkdir:
				os.mkdir(lmdb_path)
			else:
				raise Exception(f'please make sure {lmdb_path} is a valid directory')

		self._stim_id = stimulus_id
		self._lmdb_path = lmdb_path
		self.num_frames_per_clips = temporal_depth

		self.BATCH_SIZE = batch_size
		self.GPU_CNT = gpu_count

		self.gpus = []

		self.MAX_RECORDS = max_num_records # 60K max number of records per lmdb (arbitrarily chosen)
		self.MIN_RECORDS_MULT = min_records_factor # used to make sure last file is not too large (arbitrarily chosen)

		self.fpv = fpv
		self.video_start_offset = video_strt_offset
		self.clips_overlap = clips_overlap
		self.list_lmdb_meta = []

		self.units = 1

		self.video_lmdb_paths = None
		self.uneven_db = True
		self.gpu_batch_combo = None

		self.clips_dir = f'{stimulus_id}_{self.num_frames_per_clips}_{self.clips_overlap}'
		if not os.path.isdir(self.clips_dir):
			os.mkdir(self.clips_dir)
		self.clips_lmdb_data_path = f'{self._lmdb_path}/{self.clips_dir}'


	def make_from_paths(self, stimuli_paths):
		self.video_paths = stimuli_paths
		self.vid_cnt = len(self.video_paths)
		lmdb_metas = glob(f'{self.clips_lmdb_data_path}/lmdb_meta_*.csv')
		# make existence check:
		if len(lmdb_metas) > 0:
			# 
			vid_list = set(self.video_paths)
			created_metas = set()
			for i in range(len(lmdb_metas)):
				with open(f'{self.clips_lmdb_data_path}/lmdb_meta_{i}.csv') as f:
					df = pd.read_csv(f)
				created_metas.add(set(df['org_video'].values.flatten()))
			if created_metas == vid_list:
				self.video_lmdb_paths = glob(f'{self.clips_lmdb_data_path}/lmdb_*_db')
			else:
				raise Exception(f'Stimulus id {self._stim_id} does not match the videos in the LMDB')
		else:
			if not self.write_lmdb_meta():
				raise Exception('writing stimulus lmdb metas failed')
			else:
				self._create_video_dbs()


	def write_lmdb_meta(self):
		
		num_clips, start_frms = self._start_frames()
		
		db_starts, db_strides = self._records_per_meta(num_clips)
		file_strides = [int(i/num_clips) for i in db_strides[1:]]
		file_starts = [int(i/num_clips) for i in db_starts]
		
		sub_paths = [self.video_paths[offset:offset+stride] for offset, stride in zip(file_starts, file_strides)]
		
		write_data = [[[  data[i]
						  , 0 # labels is None? hacs_action_dict[os.path.basename(os.path.dirname(data[i]))]
						  , start_frms[clip_idx]
						  , num_clips*i + clip_idx + db_starts[idx]]
						for i in range(len(data)) for clip_idx in range(num_clips)]
						for idx, data in enumerate(sub_paths)]

		self.uneven_db = file_strides[-1] == file_strides[0]
		# self.units = num_clips
		return self._write_lmdb_meta(write_data)


	def _write_lmdb_meta(self, write_data):
		for group, group_paths in enumerate(write_data):
			with open(f'{self.clips_lmdb_data_path}/lmdb_meta_{group}.csv', 'w') as f:
				writer = csv.writer(f)
				writer.writerow(['org_video', 'label', 'start_frm', 'video_id'])
				writer.writerows(group_paths)
				self.list_lmdb_meta.append(f'{self.clips_lmdb_data_path}/lmdb_meta_{group}.csv')
		return True


	def _start_frames(self):
		'''
		calculate how many examples given CLIPs type:
		FULL: number of clips per video == 1
		CLIPs_ONE: each clip strides by 1, overlaping 15 frames between adjacent CLIPs
		CLIPs_TEN: overlaping 10 frames between adjacent CLIPs
		
		num_clips = ceil( (total_frames_per_video - temporal_depth - offset) / clips_stride ) + 1
				 
		given num_clips per video, calculate frame starts for videos:
			start_frm[0] = (total_frames_per_video - temporal_depth) - stride * (num_clips - 1)
			start_frm[i] = start_frm[i-1] + 6
		'''
		video_width = (self.fpv - self.video_start_offset) # 60
		clips_stride = (self.num_frames_per_clips - self.clips_overlap) 
		num_CLIPS = ceil((video_width - self.num_frames_per_clips)/clips_stride) + 1
		initial_frame = (self.fpv - self.num_frames_per_clips) - (num_CLIPS - 1) * clips_stride
		start_frms = [initial_frame + i*clips_stride for i in range(num_CLIPS)]

		assert all(start_frms[i] > 0 for i in range(len(start_frms)))
		assert any(start_frms[i] <= self.video_start_offset for i in range(len(start_frms)))
		return num_CLIPS, start_frms


	def _records_per_meta(self, num_clips):
		"""
		Caffe2 video model does not pad batched data
		this utility function will distribute batched data into even number of record files 
		a multiple of NUM_GPU and BATCH_SIZE

		the remainder will be added to a final meta file with a minimum of 
		total video remainder * MIN_RECORDS_MULT records

		returns list of where in video_paths list lmdb should begin creating DB
		and a list of how many videos in list it should consume
		"""

		total_num_records = num_clips * self.vid_cnt
		div_criteria = num_clips * self.BATCH_SIZE * self.GPU_CNT # extract_features requires number of records to divide evenly
		# start with 1 files:
		num_files = 1
		# files_rem = int(total_num_records%num_files)
		records_per_file = int(total_num_records/num_files)
		
		if records_per_file > self.MAX_RECORDS:
			# files_rem = int(total_num_records % self.MAX_RECORDS)
			num_files = int(total_num_records / self.MAX_RECORDS)
			records_per_file = int(total_num_records/num_files)
			
		rem_per_file = int(records_per_file % div_criteria)
		records_per_file = records_per_file - rem_per_file
		
		file_starts = [int(records_per_file*i) for i in range(0,num_files+1)]
		file_strides = [int(records_per_file) for i in range(num_files)]
		
		rem_total = total_num_records - num_files * records_per_file
		temp_rem = rem_total
		if rem_total > div_criteria * self.MIN_RECORDS_MULT:
			temp_rem = int(rem_total % div_criteria)
			extra_file = rem_total - temp_rem
			file_starts.append(int(extra_file + file_starts[-1]))
			file_strides.append(int(extra_file))
			num_files+=1

		if temp_rem > 0:
			file_starts.append(int(file_starts[-1] + file_strides[-1]))
			file_strides.append(int(temp_rem))
			self._get_gpu_batch_combo(temp_rem)

		assert all(file_starts[i]%div_criteria == 0 for i in range(1,num_files))
		assert total_num_records - file_starts[-1] == temp_rem
		assert sum(file_strides) == total_num_records

		return file_starts, file_strides

	def _get_gpu_batch_combo(self, file_remainder):
		gpu_check = int(file_remainder % self.GPU_CNT)
		batch_check = int(file_remainder % self.BATCH_SIZE)
		if gpu_check==0 and (batch_check==0 or batch_check!=0):
			self.gpu_batch_combo = [self.GPU_CNT, 1]
		elif gpu_check!=0 and batch_check==0:
			self.gpu_batch_combo = [1, self.BATCH_SIZE]
		else:
			self.gpu_batch_combo = [1, 1]

	def _create_video_dbs(self):
		"""
			create_video_db(
				args.list_file,
				args.output_file,
				args.use_list,
				args.use_video_id,
				args.use_start_frame,
				args.num_epochs
			)
		"""
		use_list = 1
		use_video_id = 1
		use_start_frame = 1

		list_lmdb_output = [f'{self.clips_lmdb_data_path}/lmdb_{i}_db' for i in range(len(self.list_lmdb_meta))]

		for i in range(len(self.list_lmdb_meta)):
			create_video_db(list_file=self.list_lmdb_meta[i], output_file=list_lmdb_output[i], use_list=use_list, use_video_id=use_video_id, use_start_frame=use_start_frame)

		self.video_lmdb_paths = list_lmdb_output
