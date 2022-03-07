import torch
import torchac
from utils.helpers import *
from torchvision import transforms

def pmf_to_cdf(pmf):
  cdf = pmf.cumsum(dim=-1)
  spatial_dimensions = pmf.shape[:-1] + (1,)
  zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)

  cdf = torch.cat([zeros, cdf], dim=-1)
  return cdf

def encode_padding(padding, img_name, ENC_DIR):
    # Encode padding
    pad_h = padding[0]
    pad_w = padding[1]

    pad_numpy = [pad_w, pad_h]
    pad_bytes = bytes(pad_numpy)

    filename = ENC_DIR + img_name + '/padding.bin'

    with open(filename, 'wb') as fout:
        fout.write(pad_bytes)    

def encode_jpegxl(img_a, img_name, img_name_wo_space, H, W, ENC_DIR):

    start_time = time()

    img_a = torch.unsqueeze(img_a, dim=2)

    img_a = (img_a.squeeze()).permute(1,2,0)
    img_a = img_a.numpy()
    img_a = YUV2RGB(img_a)

    img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2BGR)
    
    savename = img_name + '.png'

    cv2.imwrite(savename, img_a)

    os.system('jpegxl/build/tools/cjxl "%s" %s -q 100' % (savename, ENC_DIR + img_name_wo_space + '/jpegxl.jxl'))

    filesize = os.stat(ENC_DIR + img_name_wo_space + '/jpegxl.jxl').st_size
    
    end_time = time()

    jpegxl_time = end_time - start_time

    os.system('rm "%s"' % (savename))

    jpegxl_bpp = 8*filesize / (H*W)

    return jpegxl_bpp, jpegxl_time

def encode_convert_to_int_and_normalize(cdf_float, sym, check_input_bounds=False):
  if check_input_bounds:
    if cdf_float.min() < 0:
      raise ValueError(f'cdf_float.min() == {cdf_float.min()}, should be >=0.!')
    if cdf_float.max() > 1:
      raise ValueError(f'cdf_float.max() == {cdf_float.max()}, should be <=1.!')
    Lp = cdf_float.shape[-1]
    if sym.max() >= Lp - 1:
      raise ValueError
  cdf_int = _convert_to_int_and_normalize(cdf_float, True)
  return cdf_int

def decode_convert_to_int_and_normalize(cdf_float, needs_normalization=True):
    cdf_int = _convert_to_int_and_normalize(cdf_float, needs_normalization)
    return cdf_int

def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    PRECISION=16
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(
        2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
        cdf.add_(r)
    return cdf    

def encode_torchac(pmf_softmax, q_res, mask, img_name, color, loc, H, W, ENC_DIR, EMPTY_CACHE, frequency='low'):
    
    pmf_softmax = pmf_softmax.permute(0,2,3,1)
    pmf_softmax = torch.unsqueeze(pmf_softmax, dim=1)

    cdf = pmf_to_cdf(pmf_softmax)

    if EMPTY_CACHE:
        del pmf_softmax
        torch.cuda.empty_cache()

    cdf = cdf.clamp(max=1.)

    sym = q_res.to(torch.int16)

    _,_,h,w,c = cdf.shape

    cdf = cdf.reshape(1,1,h*w,c)
    sym = sym.reshape(1,1,h*w)

    encode_idx = (mask.flatten()==1).nonzero(as_tuple=False)

    if encode_idx.nelement()==0:
        byte_stream = str.encode('None')
    else:
        cdf_encode = cdf[:,:,encode_idx,:]
        sym_encode = sym[:,:,encode_idx]

        cdf_int = encode_convert_to_int_and_normalize(cdf_encode, sym_encode, check_input_bounds=True)

        byte_stream = torchac.encode_int16_normalized_cdf(cdf_int.detach().cpu(), sym_encode.detach().cpu())

    if frequency=='low':
        f_name = 'L'
    else:
        f_name = 'H'

    filename = ENC_DIR + img_name + '/' + color + '_' + loc + '_' + f_name + '.bin'

    with open(filename, 'wb') as fout:
        fout.write(byte_stream)
    fout.close()

    filesize = os.stat(filename).st_size

    bpp = 8*filesize / (H*W)

    return bpp

def decode_padding(ENC_DIR, img_name):

    filename = ENC_DIR + img_name + '/padding.bin'

    with open(filename, 'rb') as fin:
        pad_bytes = fin.read()
    
    return list(pad_bytes)

def decode_jpegxl(ENC_DIR, img_name):

    tensor_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    # Decode jpegxl
    jpegxl_name = ENC_DIR + img_name + '/jpegxl.jxl'

    os.system('jpegxl/build/tools/djxl "%s" %s' % (jpegxl_name, 'output.png'))

    # img_a read
    img_a = cv2.imread('output.png')
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)

    os.system('rm "%s"' % ('output.png'))

    img_a = RGB2YUV(img_a)
    img_a = tensor_transform(img_a)    

    return img_a

def decode_torchac(pmf_softmax, img_name, mask, color, loc, ENC_DIR, EMPTY_CACHE, frequency='low'):

    if frequency=='low':
        f_name = 'L'
    else:
        f_name = 'H'

    filename = ENC_DIR + img_name + '/' + color + '_' + loc + '_' + f_name + '.bin'

    with open(filename, 'rb') as fin:
        byte_stream = fin.read()
    fin.close()

    pmf_softmax = pmf_softmax.permute(0,2,3,1)
    pmf_softmax = torch.unsqueeze(pmf_softmax, dim=1)

    cdf = pmf_to_cdf(pmf_softmax)

    if EMPTY_CACHE:
        del pmf_softmax
        torch.cuda.empty_cache()

    cdf = cdf.clamp(max=1.)

    _,_,h,w,c = cdf.shape

    cdf = cdf.reshape(1,1,h*w,c)

    decode_idx = (mask.flatten()==1).nonzero(as_tuple=False)
    if decode_idx.nelement()==0:
        decoded_sym_freq = var_or_cuda(torch.zeros_like(mask, dtype=torch.int16))

    else:
        cdf_decode = cdf[:,:,decode_idx,:]

        cdf_int = decode_convert_to_int_and_normalize(cdf_decode, True)

        decoded_sym = torchac.decode_int16_normalized_cdf(cdf_int.detach().cpu(), byte_stream)

        decoded_sym = torch.squeeze(var_or_cuda(decoded_sym))

        decoded_sym_freq = var_or_cuda(torch.zeros_like(mask, dtype=torch.int16))

        decode_idx = torch.nonzero(mask, as_tuple=False)
        decode_idx = torch.transpose(decode_idx, 1, 0)
        decode_idx = [elem for elem in decode_idx]

        decoded_sym_freq = decoded_sym_freq.index_put(decode_idx, decoded_sym)

    return decoded_sym_freq

def read_img(img_name):

    tensor_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    img = cv2.cvtColor(cv2.imread(img_name),cv2.COLOR_BGR2RGB)

    ori_img = img

    img = RGB2YUV(img)
    img = tensor_transform(img)

    img, padding = pad_img(img)
    img = space_to_depth_tensor(img)

    img_a, img_b, img_c, img_d = img[:,0], img[:,1], img[:,2], img[:,3]
    
    img_a, img_b, img_c, img_d = torch.unsqueeze(img_a, dim=0), torch.unsqueeze(img_b, dim=0), torch.unsqueeze(img_c, dim=0), torch.unsqueeze(img_d, dim=0)

    return img_a, img_b, img_c, img_d, ori_img, img_name, padding

def abcd2img(imgs, color_names):

    # loc_imgs{['a', 'd', 'b', 'c']}
    loc_imgs = {}

    for loc in ['a','d','b','c']:
        temp_img = torch.Tensor()
        temp_img = var_or_cuda(temp_img)

        for color in color_names:
            temp_img = torch.cat((temp_img, imgs[color][loc]), dim=1)
        
        loc_imgs[loc] = temp_img

    # loc_imgs to whole image
    output_img = torch.Tensor()
    output_img = var_or_cuda(output_img)

    for loc in ['a','b','c','d']:
        output_img = torch.cat((output_img, loc_imgs[loc]), dim=0)

    # output_img (Tensor) to output_img (numpy)
    output_img = output_img.permute(1,0,2,3)
    output_img = torch.unsqueeze(output_img, dim=0)

    output_img = depth_to_space_tensor(output_img, BLOCK_SIZE=2)
    
    output_img = torch.squeeze(output_img)
    output_img = output_img.permute(1,2,0)
    output_img = output_img.detach().cpu().numpy()
    output_img = YUV2RGB(output_img)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    return output_img

def modify_imgname(img_name):

    name_split = img_name.split(".")
    img_name = img_name.replace('.' + name_split[-1], '')

    name_split = img_name.split("/")
    img_name = name_split[-1]
    
    return img_name