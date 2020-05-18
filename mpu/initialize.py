# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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


"""Model and data parallel groups."""


def initialize_model_parallel(model_parallel_size_):
    raise RuntimeError(
        'Use fairseq.distributed_utils.initialize_model_parallel instead'
    )


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    raise NotImplementedError


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    raise RuntimeError(
        'Use fairseq.distributed_utils.get_model_parallel_group instead'
    )


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    raise RuntimeError(
        'Use fairseq.distributed_utils.get_data_parallel_group instead'
    )


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    raise RuntimeError(
        'Use fairseq.distributed_utils.get_model_parallel_world_size instead'
    )


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    raise RuntimeError(
        'Use fairseq.distributed_utils.get_model_parallel_rank instead'
    )


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zeor
    in the model parallel group."""
    raise NotImplementedError


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    raise RuntimeError(
        'Use fairseq.distributed_utils.get_data_parallel_world_size instead'
    )


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    raise RuntimeError(
        'Use fairseq.distributed_utils.get_data_parallel_rank instead'
    )


def destroy_model_parallel():
    """Set the groups to none."""
    raise NotImplementedError
