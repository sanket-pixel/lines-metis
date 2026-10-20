// SPDX-License-Identifier: AGPL-3.0-or-later
#include "inference/aipu_context.h"

#include <axruntime/axruntime.h>
#include <stdexcept>

namespace lines {
AipuContext::AipuContext() : context_(axr_create_context()) {
    if (!context_) {
        throw std::runtime_error("AipuContext: cannot create AxRuntime context");
    }
}

AipuContext::~AipuContext() {
    axr_destroy(reinterpret_cast<const axrObject*>(context_));
}

axrContext* AipuContext::get() const {
    return context_;
}
}
