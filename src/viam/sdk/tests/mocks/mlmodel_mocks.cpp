// Copyright 2023 Viam Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <viam/sdk/tests/mocks/mlmodel_mocks.hpp>

namespace viam {
namespace sdktests {

sdk::MLModelService::infer_result MockMLModelService::infer(const tensor_map& inputs) {
    return {};
}

MockMLModelService& MockMLModelService::metadata(struct metadata metadata) {
    metadata_ = std::move(metadata);
    return *this;
}

struct sdk::MLModelService::metadata MockMLModelService::metadata() {
    return metadata_;
}

}  // namespace sdktests
}  // namespace viam