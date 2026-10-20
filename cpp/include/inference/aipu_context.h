// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

struct axrContext;

namespace lines {
// Share one context across models; it must outlive every attached backend.
class AipuContext {
public:
    AipuContext();  // axr_create_context; throws on failure.
    ~AipuContext();  // axr_destroy after all model backends are destroyed.
    AipuContext(const AipuContext&) = delete;
    AipuContext& operator=(const AipuContext&) = delete;
    AipuContext(AipuContext&&) = delete;
    AipuContext& operator=(AipuContext&&) = delete;

    axrContext* get() const;

private:
    axrContext* context_ = nullptr;
};
}
