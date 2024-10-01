from enum import Enum
import numpy as np

from luma.interface.typing import TensorLike


class MergeMode(Enum):
    CHCAT = "chcat"
    SUM = "sum"
    HADAMARD = "hadamard"
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    DOT = "dot"
    SUB = "sub"

    def __init__(self, value: str):
        self.reduce_func: callable = np.mean
        _ = value

    def forward(self, f_queue: list[TensorLike]) -> TensorLike:
        match self:
            case MergeMode.CHCAT:
                return np.concatenate(f_queue, axis=1)

            case MergeMode.SUM:
                return np.sum(f_queue, axis=0)

            case MergeMode.HADAMARD:
                X = np.ones_like(f_queue[0])
                for tensor in f_queue:
                    X *= tensor
                return X

            case MergeMode.AVG:
                return np.mean(f_queue, axis=0)

            case MergeMode.MAX:
                stacked = np.stack(f_queue, axis=0)
                return np.max(stacked, axis=0)

            case MergeMode.MIN:
                stacked = np.stack(f_queue, axis=0)
                return np.min(stacked, axis=0)

            case MergeMode.DOT:
                return np.dot(f_queue[0], f_queue[1])

            case MergeMode.SUB:
                result = f_queue[0]
                for tensor in f_queue[1:]:
                    result -= tensor
                return result

    def backward(
        self, f_queue: list[TensorLike], d_out: TensorLike, i: int
    ) -> TensorLike:
        d_out_i = None
        match self:
            case MergeMode.CHCAT:
                cum_ch = [0]
                for tensor in f_queue:
                    cum_ch.append(cum_ch[-1] + tensor.shape[1])
                d_out_i = d_out[:, cum_ch[i] : cum_ch[i + 1], ...]

            case MergeMode.SUM:
                d_out_i = d_out

            case MergeMode.HADAMARD:
                prod_except_current = np.ones_like(f_queue[0])
                for j in range(len(f_queue)):
                    if j != i:
                        prod_except_current *= f_queue[j]
                d_out_i = d_out * prod_except_current

            case MergeMode.AVG:
                d_out_i = d_out / len(f_queue)

            case MergeMode.MIN | MergeMode.MAX:
                stacked = np.stack(f_queue, axis=0)
                merged = np.max(stacked, axis=0)
                mask = f_queue[i] == merged

                total_mask = np.sum(
                    [tensor == merged for tensor in f_queue],
                    axis=0,
                )
                total_mask = np.clip(total_mask, a_min=1, a_max=None)

                grad = (d_out * mask / total_mask).astype(d_out.dtype)
                d_out_i = grad

            case MergeMode.DOT:
                if i == 0:
                    d_out_i = np.dot(d_out, f_queue[1].T)
                elif i == 1:
                    d_out_i = np.dot(f_queue[0].T, d_out)

            case MergeMode.SUB:
                d_out_i = d_out if i == 0 else -d_out

        if np.prod(f_queue[i].shape) < np.prod(d_out_i.shape):
            d_out_i = self._inverse_broadcast(d_out_i, f_queue[i].shape)

        elif np.prod(f_queue[i].shape) > np.prod(d_out_i.shape):
            d_out_i = np.broadcast_to(d_out_i, f_queue[i].shape)
        else:
            pass

        if f_queue[i].shape != d_out_i.shape:
            raise ValueError(
                f"Failed to broadcast {d_out_i.shape} to its"
                + f" corresponding forward shape {f_queue[i].shape}"
            )

        return d_out_i

    def _inverse_broadcast(
        self, tensor: TensorLike, inverse_shape: tuple[int]
    ) -> TensorLike:
        ten_shape = tensor.shape
        len_diff = len(ten_shape) - len(inverse_shape)

        if len_diff > 0:
            inverse_shape = (1,) * len_diff + inverse_shape
        elif len_diff < 0:
            raise ValueError(
                "Original shape has more dimensions than the broadcasted array."
            )

        axes_to_reduce = []
        for axis, (org_dim, brd_dim) in enumerate(zip(inverse_shape, ten_shape)):
            if org_dim == 1 and brd_dim > 1:
                axes_to_reduce.append(axis)

        if not axes_to_reduce:
            return tensor

        reduced_array = self.reduce_func(
            tensor,
            axis=tuple(axes_to_reduce),
            keepdims=True,
        )
        if len_diff > 0:
            reduced_array = reduced_array.reshape(inverse_shape)

        return reduced_array
