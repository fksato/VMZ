from setuptools import setup, find_packages

requirements = [
	"tqdm",
]

setup(
	name='vmz_interface',
	version='0.1.0',
	packages=find_packages(exclude=['tests']),
	url='https://github.com/fksato/VMZ',
	author='Fukushi Sato',
	author_email='f.kazuo.sato@gmail.com',
	description='Facebook VMZ interface',
	install_requires=requirements,
	classifiers=[
			        'Development Status :: Pre-Alpha',
			        'Intended Audience :: Developers',
			        'License :: Apache License',
			        'Natural Language :: English',
			        'Programming Language :: Python :: 3.6',
			        'Programming Language :: Python :: 3.7',
			    ],
	# entry_points={
	# 			'console_scripts':[
	# 								'extract_features = tools.extract_features:main',
	# 								'db_video_create = data.db_video_create:main'
	# 					          ]

	# 			}
)
