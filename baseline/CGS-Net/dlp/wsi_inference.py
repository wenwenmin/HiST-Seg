import os.path
import tiffslide
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # might need this we looked at L format and it should be able to hold the large file sizes
# they have a max_pixel limit for decompression bombs. there should be no issue removing the limit.
from einops import rearrange
import wsi_inference_utils as ws_utils
import numpy as np
from models.model import ModelMaker
from empatches import EMPatches
from datetime import datetime
emp = EMPatches()
import tifffile


def wsi_inference(testwsi_fp):
    # torch.set_num_threads(7)
    # pytorch will automatically use all available threads, so when multithreading model inference. we need to make sure that each thread
    # only uses some many cpus otherwise the threads will fight over the cpus.
    model_obj = ModelMaker(model_params=model_options)
    model = model_obj.make_model()
    model_obj.print_model_params(model=model)
    model.to('cpu')
    model.eval()
    # need to update this code if doing a single input model
    test_transform = model_obj.make_val_transform()
    test_transform.add_targets({'image1': 'image'})
    print("model made", flush=True)

    slidename = os.path.split(testwsi_fp)[-1]
    print(f"working on slide {slidename}",flush=True)
    slide = tiffslide.open_slide(testwsi_fp)  # this opens the slide but we need to close it
    slide_map = np.float32(slide.get_thumbnail(slide.level_dimensions[level]))
    # slide_map = np.float32(slide.get_thumbnail(slide.level_dimensions[level]))[:4000,:10000,:] # use for debugging
    tissue_slide_mask = ws_utils.get_slide_tissue_mask(slidename, slide_mask_path)

    img_patches, indices = emp.extract_patches(slide_map, patchsize=224, overlap=0.80, stride=56)
    print("56 is new stride ", flush=True)
    print("0.80 is new overlap ", flush=True)


    print(f"number of patches from {slidename}",len(img_patches),flush=True)
    print(f"num indices {slidename}", len(indices), flush=True)

    slide_map_lv3 = np.float32(slide.get_thumbnail(slide.level_dimensions[level+1]))  # rgb slide

    print("shape of slide_map_lv3 is: ", slide_map_lv3.shape, flush=True)
    slide.close()
    del slide_map

    lv3_prep = [(x+112,y+112) for y,yoff,x,xoff in indices]
    lv3_indices = [ws_utils.get_indices_lowerlvl(current_center_cords=cords, downsample_current=resolution_levels[level], downsample_factor_lower_level=resolution_levels[level+1]) for cords in lv3_prep]
    output_patches = []
    border_width = 20
    with torch.no_grad():
        for idx,lvl2_patch in enumerate(img_patches):

            ci, cj = indices[idx][2] + 112, indices[idx][0] + 112
            tissue_ratio = ws_utils.return_ratios(ci=ci, cj=cj, tissue_mask=tissue_slide_mask)
            if tissue_ratio < 0.05:
                # we are doing a pass on the indices and removing any indices (for a patch) that contain less than x% tissue
                # in other words we are removing the background patches from the test slides
                # pop over list and need index remember the list re-indexes
                myzeros = np.zeros((224,224))
                output_patches.append(myzeros)
                continue

            lvl3_patch = slide_map_lv3[int(lv3_indices[idx][1]-112):int(lv3_indices[idx][1]+112), int(lv3_indices[idx][0]-112):int(lv3_indices[idx][0]+112)]
            lv2_np = np.array(lvl2_patch)
            empty_mask = np.zeros(lv2_np.shape)

            augmentations = test_transform(image=lv2_np, image1=lvl3_patch, mask=empty_mask)
            image_detail = augmentations["image"]
            image_context = augmentations["image1"]
            # format tensor the way pytorch likes
            image_context = rearrange(image_context, 'c h w -> 1 c h w')
            image_detail = rearrange(image_detail, 'c h w -> 1 c h w')


            lvl2_pred =  torch.sigmoid(model(image_detail,image_context))
            lvl2_pred = rearrange(lvl2_pred, 'b c h w -> (b c h) w')
            lv2_pred_np = lvl2_pred.numpy()

            lv2_pred_np[:border_width, :] = 0 #top
            lv2_pred_np[-border_width:, :] = 0 #bottom
            lv2_pred_np[:, :border_width] = 0 #left
            lv2_pred_np[:, -border_width:] = 0 # right

            # in case we get very small probabilities need to set them to zero for image saving reasons
            lv2_pred_np_f16 = lv2_pred_np.astype(np.float16)

            output_patches.append(lv2_pred_np_f16)

            if idx % 100 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"% complete for {slidename}",len(output_patches)/len(img_patches) * 100, current_time, flush=True)


    del lv3_indices, lv3_prep, img_patches, model, slide_map_lv3
    # Merge the processed patches
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("before merge patches: ", current_time, flush=True)
    prediction_slide_mask = emp.merge_patches(output_patches, indices, mode='max')
    del output_patches, indices
    # print("sum after emp patches ", prediction_slide_mask.sum())
    # emp likes numpy arrays much better than pytorch tensors. also note that when you want all black for a prediction, you should not remove the indices from the list
    # pop, rather skip over the bad inputs in your model prediction loop, and then make your model prediction zeros. If you pop the indices and only loop over a subset
    # then emp will not have the full dimensions of the wsi slide and you will get indexing errors. The over all flow is, that you generate the patches and indices and then loop over them, then
    # do things to the patches and save the outputs and then do the merge. Notice, I'm not removing anything.
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("after merge patches and before save: ", current_time, flush=True)
    try:
        pred_pil = Image.fromarray(prediction_slide_mask)
        pred_pil.save(f"./pred_slides/{slidename}")
    except Exception:
        prediction_slide_mask_tiff = np.array(prediction_slide_mask, dtype=np.float16) #This was an error I had np.unit8 here and the
        # the images were all black and should be a float
        tifffile.imwrite(f"./pred_slides/tiff_{slidename}", data=prediction_slide_mask_tiff)
    finally:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("after save: ", current_time, flush=True)
        # Note you need to keep the print statement helps for debugging when multithreading; depending on the issue, the code may or may not error out
        # this helps to determine where the code stopped running.
        return



if __name__ == '__main__':
    slide_mask_path = "/path/mask_lv2"
    level = 2  # needs to be an integer
    resolution_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


    slidelist = ['/path/normaltest_005.tif',
                 '/path/normaltest_070.tif',
                 '/path/normaltest_115.tif']

    model_options = {
        "model_name": "cgs_mit_b1",
        "weights": "trainval_cgs",
        "dropout": 0.15,
        "input_img_size": 224,
        "aug_type": None,
    }

    for slide in slidelist:
        wsi_inference(testwsi_fp=slide)
