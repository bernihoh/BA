# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

from stylegan2.dnnlib import submission

from stylegan2.dnnlib.submission.run_context import RunContext

from stylegan2.dnnlib.submission.submit import SubmitTarget
from stylegan2.dnnlib.submission.submit import PathType
from stylegan2.dnnlib.submission.submit import SubmitConfig
from stylegan2.dnnlib.submission.submit import submit_run
from stylegan2.dnnlib.submission.submit import get_path_from_template
from stylegan2.dnnlib.submission.submit import convert_path
from stylegan2.dnnlib.submission.submit import make_run_dir_path

from stylegan2.dnnlib.util import EasyDict

submit_config: SubmitConfig = None # Package level variable for SubmitConfig which is only valid when inside the run function.
