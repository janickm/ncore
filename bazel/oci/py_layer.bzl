# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"Wrapper macro to make three separate layers for python applications"

load("@aspect_bazel_lib//lib:tar.bzl", "mtree_spec", "tar")
load("@rules_oci//oci:defs.bzl", "oci_image")

# match *only* external repositories that have the string "python"
# e.g. this will match
#   `/hello_world/hello_world_bin.runfiles/rules_python~0.21.0~python~python3_9_aarch64-unknown-linux-gnu/bin/python3`
# but not match
#   `/hello_world/hello_world_bin.runfiles/_main/python_app`
PY_INTERPRETER_REGEX = "\\.runfiles/.*python.*-.*"

# match *only* external pip like repositories that contain the string "site-packages"
SITE_PACKAGES_REGEX = "\\.runfiles/.*/site-packages/.*"

def py_layers(name, binary, tags, compress):
    """Create three layers for a py_binary target: interpreter, third-party dependencies, and application code.

    This allows a container image to have smaller uploads, since the application layer usually changes more
    than the other two.

    Args:
        name: prefix for generated targets, to ensure they are unique within the package
        binary: a py_binary target
        tags: a list of tags to be applied to the generated targets
        compress: the compression algorithm to use for the tar files

    Returns:
        a list of labels for the layers, which are tar files
    """

    # Produce layers in this order, as the app changes most often
    layers = ["interpreter", "packages", "app"]

    # Produce the manifest for a tar file of our py_binary, but don't tar it up yet, so we can split
    # into fine-grained layers for better docker performance.
    mtree_spec(
        name = name + ".mf",
        srcs = [binary],
    )

    native.genrule(
        name = name + ".interpreter_tar_manifest",
        srcs = [name + ".mf"],
        outs = [name + ".interpreter_tar_manifest.spec"],
        cmd = "grep -v '{}' $< | grep '{}' >$@".format(SITE_PACKAGES_REGEX, PY_INTERPRETER_REGEX),
    )

    native.genrule(
        name = name + ".packages_tar_manifest",
        srcs = [name + ".mf"],
        outs = [name + ".packages_tar_manifest.spec"],
        cmd = "grep '{}' $< >$@".format(SITE_PACKAGES_REGEX),
    )

    # Any lines that didn't match one of the two grep above
    native.genrule(
        name = name + ".app_tar_manifest",
        srcs = [name + ".mf"],
        outs = [name + ".app_tar_manifest.spec"],
        cmd = "grep -v '{}' $< | grep -v '{}' >$@".format(SITE_PACKAGES_REGEX, PY_INTERPRETER_REGEX),
    )

    result = []
    for layer in layers:
        layer_target = "{}.{}_layer".format(name, layer)
        result.append(layer_target)
        tar(
            name = layer_target,
            srcs = [binary],
            mtree = "{}.{}_tar_manifest".format(name, layer),
            tags = tags,
            compress = compress,
        )

    return result

def py_oci_image(name, binary, tars = [], tags = [], compress = "gzip", **kwargs):
    "Wrapper around oci_image that splits the py_binary into layers."
    oci_image(
        name = name,
        tars = tars + py_layers(name, binary, tags, compress),
        tags = tags,
        **kwargs
    )
