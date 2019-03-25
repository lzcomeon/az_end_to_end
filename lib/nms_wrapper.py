
from lib.config import opt
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if opt.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=opt.GPU_ID)
    else:
        return cpu_nms(dets, thresh)
