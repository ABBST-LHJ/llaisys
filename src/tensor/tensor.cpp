#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#define EXCEPTION_INVALID_SHAPE(msg) throw std::invalid_argument("Invalid shape: " + std::string(msg))
#define EXCEPTION_INVALID_DIM(msg) throw std::out_of_range("Invalid dimension: " + std::string(msg))
#define EXCEPTION_INVALID_ORDER(msg) throw std::invalid_argument("Invalid permute order: " + std::string(msg))
#define EXCEPTION_INCOMPATIBLE_VIEW(msg) throw std::invalid_argument("Incompatible view: " + std::string(msg))
namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    const auto& shape_ = this->shape();
    const auto& strides_ = this->strides();
    size_t ndim_ = this->ndim();
    
    // 1. 空张量、一维张量默认连续
    if (ndim_ <= 1) {
        return true;
    }
    
    // 2. 从最后一个维度向前验证步长规律
    ptrdiff_t expected_stride = 1; // 最后一个维度的步长必须为 1
    for (size_t i = ndim_; i > 0; --i) {
        size_t dim = i - 1; // 从最后一个维度（ndim_-1）遍历到第 0 个维度
        
        // 校验当前维度的步长是否等于预期步长
        if (strides_[dim] != expected_stride) {
            return false;
        }
        
        // 计算上一个维度的预期步长（仅当不是第 0 个维度时）
        if (dim > 0) {
            expected_stride *= shape_[dim];
        }
    }
    
    // 3. 所有维度步长均满足规律，返回连续
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim_ = this->ndim();
    
    // 1. 校验 order 合法性
    if (order.size() != ndim_) {
        EXCEPTION_INVALID_ORDER("Order size (" + std::to_string(order.size()) + ") does not match tensor ndim (" + std::to_string(ndim_) + ")");
    }
    
    std::vector<bool> dim_used(ndim_, false);
    for (size_t dim : order) {
        if (dim >= ndim_) {
            EXCEPTION_INVALID_ORDER("Dimension " + std::to_string(dim) + " is out of range (0 ~ " + std::to_string(ndim_-1) + ")");
        }
        if (dim_used[dim]) {
            EXCEPTION_INVALID_ORDER("Dimension " + std::to_string(dim) + " is duplicated in order");
        }
        dim_used[dim] = true;
    }
    
    // 2. 重新排列 shape 和 strides
    TensorMeta new_meta = this->_meta;
    new_meta.shape.resize(ndim_);
    new_meta.strides.resize(ndim_);
    
    for (size_t i = 0; i < ndim_; ++i) {
        size_t original_dim = order[i];
        new_meta.shape[i] = this->_meta.shape[original_dim];
        new_meta.strides[i] = this->_meta.strides[original_dim];
    }
    
    // 3. 构造新张量（共享存储，仅修改元信息，无数据传输）
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &new_shape) const {
    // 1. 计算原始张量和新形状的元素总数
    size_t original_numel = this->numel();
    size_t new_numel = std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    
    // 2. 校验元素总数是否一致（基础兼容条件）
    if (original_numel != new_numel) {
        EXCEPTION_INCOMPATIBLE_VIEW("Element count mismatch: original (" + std::to_string(original_numel) + ") vs new (" + std::to_string(new_numel) + ")");
    }
    
    // 3. 空张量直接返回新形状（无数据，无需校验布局）
    if (original_numel == 0) {
        TensorMeta new_meta = this->_meta;
        new_meta.shape = new_shape;
        // 推导空张量的默认步长
        std::vector<ptrdiff_t> new_strides(new_shape.size());
        size_t stride = 1;
        for (size_t i = 1; i <= new_shape.size(); ++i) {
            new_strides[new_shape.size() - i] = stride;
            stride *= new_shape[new_shape.size() - i];
        }
        new_meta.strides = new_strides;
        return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
    }
    
    // 4. 校验原始张量是否连续（仅连续张量支持任意合法 view，不连续张量需特殊校验布局）
    // 简化实现：优先支持连续张量的 view（满足题目要求，且是行业常用实践）
    if (!this->isContiguous()) {
        // 拓展：若要支持不连续张量的 view，需校验新形状是否可通过原始步长拆分/合并推导
        // 此处先实现核心功能：仅连续张量允许 view
        EXCEPTION_INCOMPATIBLE_VIEW("Only contiguous tensors support view operation (non-contiguous tensor layout is incompatible).");
    }
    
    // 5. 推导新形状对应的连续步长（view 不改变数据布局，新张量仍为连续）
    std::vector<ptrdiff_t> new_strides(new_shape.size());
    size_t stride = 1;
    for (size_t i = 1; i <= new_shape.size(); ++i) {
        size_t dim = new_shape.size() - i;
        new_strides[dim] = stride;
        stride *= new_shape[dim];
    }
    
    // 6. 构造新张量（共享原始存储，仅修改元信息，无数据传输）
    TensorMeta new_meta = this->_meta;
    new_meta.shape = new_shape;
    new_meta.strides = new_strides;
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    size_t ndim_ = this->ndim();
    const auto& shape_ = this->shape();
    
    // 1. 校验维度合法性
    if (dim >= ndim_) {
        EXCEPTION_INVALID_DIM("Slice dim " + std::to_string(dim) + " out of range (0 ~ " + std::to_string(ndim_-1) + ")");
    }
    
    // 2. 校验 start 和 end 合法性
    size_t dim_size = shape_[dim];
    if (start > end) {
        EXCEPTION_INVALID_DIM("Slice start (" + std::to_string(start) + ") > end (" + std::to_string(end) + ")");
    }
    if (end > dim_size) {
        EXCEPTION_INVALID_DIM("Slice end (" + std::to_string(end) + ") exceeds dim size (" + std::to_string(dim_size) + ")");
    }
    if (start == end) {
        // 返回空切片张量
        TensorMeta new_meta = this->_meta;
        new_meta.shape[dim] = 0;
        return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
    }
    
    // 3. 计算新张量的 offset（切片对应的内存偏移量）
    size_t element_size = this->elementSize();
    size_t new_offset = this->_offset + (start * this->_meta.strides[dim] * element_size);
    
    // 4. 构造新张量的元信息（修改对应维度的 shape，strides 不变）
    TensorMeta new_meta = this->_meta;
    new_meta.shape[dim] = end - start;
    
    // 5. 构造新张量（共享存储，修改元信息和 offset，无数据传输）
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 1. 校验输入有效性
    if (src_ == nullptr) {
        throw std::invalid_argument("Source pointer cannot be null.");
    }
    const std::byte *src = reinterpret_cast<const std::byte *>(src_);
    
    // 2. 计算需要拷贝的总字节数
    size_t total_bytes = this->numel() * this->elementSize();
    if (total_bytes == 0) {
        return; // 空张量无需拷贝
    }
    
    // 3. 获取当前设备上下文和运行时 API
    core::context().setDevice(this->deviceType(), this->deviceId());
    auto *api = core::context().runtime().api();
    if (api == nullptr) {
        throw std::runtime_error("Failed to get device runtime API.");
    }
    
    // 4. 执行对应类型的内存拷贝（主机 -> 目标设备/主机）
    llaisysMemcpyKind_t memcpy_kind;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // 目标是 CPU：直接使用 std::memcpy 同步拷贝
        std::memcpy(this->data(), src, total_bytes);
    } else {
        // 目标是设备（如 GPU）：使用运行时 API 的 D2H 反向（H2D）同步拷贝
        memcpy_kind = LLAISYS_MEMCPY_H2D;
        api->memcpy_sync(this->data(), src, total_bytes, memcpy_kind);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys