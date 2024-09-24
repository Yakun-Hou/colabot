import copy

import numpy as np
import cv2
import torch
import torchvision
import ipdb
from imgcat import imgcat

from lib.test.tracker.data_utils import Preprocessor_wo_mask
from lib.utils.box_ops import box_xywh_to_cxcywh


def draw_bbox(img, bbox, color):
    """
    - bbox: [cx, cy, w, h] in range [0, 1]
    - color: R, G, B
    """
    im_h, im_w, _ = img.shape
    cx, cy, w, h = bbox
    x0 = int((cx - w / 2) * im_w)
    y0 = int((cy - h / 2) * im_h)
    x1 = int(x0 + w * im_w)
    y1 = int(y0 + h * im_h)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    img = cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    return img


@torch.no_grad()
def get_patch_overlay(img, patch_ids, patch_size=16):
    """
    Args:
        img: (H, W, 3)
        patch_ids: (N)

    Returns:
        overlay (np.ndarray) of shape (mask_size, mask_size, 3)
    """
    size = img.shape[0]
    n_patch_per_row = size // patch_size
    patch_h = patch_ids // n_patch_per_row
    patch_w = patch_ids % n_patch_per_row

    overlay = (torch.ones(size, size, 3) * 255).byte().numpy()
    for h, w in zip(patch_h, patch_w):
        overlay[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size, :] = \
            img[h * patch_size:(h + 1) * patch_size, w * patch_size:(w + 1) * patch_size, :]
    return overlay


def vis_attn(attn, out_size=(256, 256), image=None):
    """
    Args:
        attn: torch.tensor (H, W)
        out_size:

    Returns:
        torch.tensor (3, *out_size)
    """
    interpolation = None # cv2.INTER_NEAREST

    attn_map = attn / attn.max()
    attn_map = cv2.resize(attn_map.numpy(), out_size, interpolation=interpolation)
    attn_map = np.uint8(attn_map * 255)
    attn_map = cv2.applyColorMap(attn_map, cv2.COLORMAP_JET)
    attn_map = cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB)
    if image is not None:
        attn_map = cv2.addWeighted(attn_map, 0.5, image, 0.5, 0)
    return torch.tensor(attn_map).permute(2, 0, 1)  # (3, H, W)


def linker_vis(debug_info, ps_id, s_id, vis_layers=None, attn_weights=None, save_path=None,
               ps_ind=(0.5, 0.5), gt_bbox=None, ps_gt_bbox=None):
    """

    Args:
        template: (3, H1, W1)
        prev_search: (3, H2, W2)
        search: (3, H2, W2)
        pred_bbox: (4)
        l_prev_search_patch_ids: (L, N)
        l_search_patch_ids: (L, N)
        vis_layers:
        attn_weights: list of L (h, N, N)
        save_path:

    Returns:
    """
    template = debug_info['template']
    prev_search = debug_info['ps']
    search = debug_info['search']
    pred_bbox = debug_info['pred_bbox']
    score_map = debug_info['score_map']
    pred_score = debug_info['pred_score']
    l_ps_patch_ids = debug_info['l_ps_patch_ids']
    l_s_patch_ids = debug_info['l_s_patch_ids']

    _, _, Wt = template.shape
    Wt //= 16
    Nt = Wt ** 2
    _, _, Wps = prev_search.shape
    Wps //= 16
    Nps = Wps ** 2
    _, _, Ws = search.shape
    Ws //= 16
    Ns = Ws ** 2
    out_size = (search.shape[1], search.shape[2])
    preprocessor = Preprocessor_wo_mask()

    template_image = preprocessor.inverse_process(template)
    pad = (search.shape[1] - template.shape[1]) // 2
    template_image = cv2.copyMakeBorder(template_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, (0, 0, 0))

    prev_search_image = preprocessor.inverse_process(prev_search)
    prev_search_overlay = get_patch_overlay(prev_search_image, l_ps_patch_ids[-1])
    prev_search_image_eli = cv2.addWeighted(prev_search_image, 0.3, prev_search_overlay, 0.7, 0)
    # prev_search_image_eli = prev_search_image.copy()
    cv2.circle(prev_search_image_eli, (int(search.shape[1] * 0.5), int(search.shape[1] * 0.5)), 3,
               (0, 0, 255), -1)
    prev_search_image_eli = cv2.putText(prev_search_image_eli, '#%03d' % ps_id, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    if ps_gt_bbox is not None:
        prev_search_image_eli = draw_bbox(prev_search_image_eli, ps_gt_bbox, (0, 255, 0))

    search_image = preprocessor.inverse_process(search)
    search_overlay = get_patch_overlay(search_image, l_s_patch_ids[-1])
    search_image_eli = cv2.addWeighted(search_image, 0.3, search_overlay, 0.7, 0)
    # search_image_eli = search_image.copy()
    search_image_eli = draw_bbox(search_image_eli, pred_bbox, (255, 0, 0))
    # cv2.circle(search_image_eli, (int(search.shape[1] * 0.5), int(search.shape[1] * 0.5)), 3,
    #            (0, 0, 255), -1)
    search_image_eli = cv2.putText(search_image_eli, '#%03d' % s_id, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    if gt_bbox is not None:
        search_image_eli = draw_bbox(search_image_eli, gt_bbox, (0, 255, 0))

    if score_map is not None:
        max_score = score_map.max()
        score_map = vis_attn(score_map, out_size, search_image)
        score_map_im = score_map.permute(1, 2, 0).numpy()
        # score_map_im = cv2.putText(score_map_im, 'Max: %.2f' % float(max_score), (10, 30),
        #                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # if pred_score is not None:
        #     pred_score = torch.sigmoid(pred_score)
        #     score_map_im = cv2.putText(score_map_im, 'Score: %.2f' % float(pred_score), (10, 60),
        #                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        score_map = torch.tensor(score_map_im).permute(2, 0, 1)
    else:
        score_map = torch.zeros((3, *out_size))

    ps_center = (int(Wps * ps_ind[1]), int(Wps * ps_ind[0]))  # y, x
    attn = attn_weights[-1].mean(dim=0).squeeze().cpu()  # (Nt + 2*Ns, Nt + 2*Ns)
    l = vis_layers[-1]
    l_ps_patch_ids.insert(0, torch.arange(Nps))
    cur_Nps = l_ps_patch_ids[l].shape[0]
    ps_s_attn = torch.zeros(Ns, Nps)
    xy = torch.cartesian_prod(l_s_patch_ids[l], l_ps_patch_ids[l])  # (Ns*Ns, 2)
    ps_s_attn[xy[:, 0], xy[:, 1]] = attn[Nt + cur_Nps:, Nt:Nt + cur_Nps].reshape(-1)
    ps_s_attn = ps_s_attn.reshape(Ws, Ws, Wps, Wps)[..., ps_center[0], ps_center[1]]  # (Ws, Ws))

    im_grid = [
        torch.tensor(template_image).permute(2, 0, 1),
        torch.tensor(prev_search_image_eli).permute(2, 0, 1),
        torch.tensor(search_image_eli).permute(2, 0, 1),
        # score_map
        vis_attn(ps_s_attn, out_size, search_image)
    ]

    # # visualize attention map
    # t_center = (Wt // 2, Wt // 2)  # center of template
    # ps_center = (int(Wps * ps_ind[1]), int(Wps * ps_ind[0]))  # y, x
    # l_ps_patch_ids.insert(0, torch.arange(Nps))
    # l_s_patch_ids.insert(0, torch.arange(Ns))
    #
    # for i, l in enumerate(vis_layers):
    #     # attn_weights contains multi-head attention of L layers
    #     # calculate the average of attention of all heads
    #     attn = attn_weights[i].mean(dim=0).squeeze().cpu()  # (Nt + 2*Ns, Nt + 2*Ns)
    #     # attn = attn_weights[i][0].squeeze().cpu()  # (Nt + 2*Ns, Nt + 2*Ns)
    #
    #     # complete eliminated parts
    #     cur_Nps = l_ps_patch_ids[l].shape[0]
    #
    #     # ## template to prev_search attention
    #     # t_ps_attn = torch.zeros(Nps, Nt)
    #     # t_ps_attn[l_ps_patch_ids[l], :] = attn[Nt:Nt + cur_Nps, :Nt]
    #     # t_ps_attn = t_ps_attn.reshape(Wps, Wps, Wt, Wt)[..., t_center[0], t_center[1]]  # (Ws, Ws)
    #     # if Wps < Ws:
    #     #     pad = (Ws - Wps) // 2
    #     #     t_ps_attn = cv2.copyMakeBorder(t_ps_attn.numpy(), pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    #     #     t_ps_attn = torch.tensor(t_ps_attn)
    #     # im_grid[1].append(vis_attn(t_ps_attn, out_size, prev_search_image))
    #     #
    #     # ## template to search attention
    #     # t_s_attn = torch.zeros(Ns, Nt)
    #     # t_s_attn[l_s_patch_ids[l], :] = attn[Nt + cur_Nps:, :Nt]
    #     # t_s_attn = t_s_attn.reshape(Ws, Ws, Wt, Wt)[..., t_center[0], t_center[1]]  # (Ws, Ws)
    #     # im_grid[2].append(vis_attn(t_s_attn, out_size, search_image))
    #
    #     ## prev_search to search attention
    #     ps_s_attn = torch.zeros(Ns, Nps)
    #     xy = torch.cartesian_prod(l_s_patch_ids[l], l_ps_patch_ids[l])  # (Ns*Ns, 2)
    #     ps_s_attn[xy[:, 0], xy[:, 1]] = attn[Nt + cur_Nps:, Nt:Nt + cur_Nps].reshape(-1)
    #     ps_s_attn = ps_s_attn.reshape(Ws, Ws, Wps, Wps)[..., ps_center[0], ps_center[1]]  # (Ws, Ws))
    #     im_grid[1].append(vis_attn(ps_s_attn, out_size, search_image))

    # vis = torch.cat([torch.stack(x) for x in im_grid])
    vis = im_grid
    vis = torchvision.utils.make_grid(vis, nrow=len(im_grid))
    if save_path is None:
        imgcat(vis)
        print('')
        ipdb.set_trace()
    else:
        vis = vis.permute(1, 2, 0).numpy()[:, :, ::-1]
        cv2.imwrite(save_path, vis)


def linker_t_vis(debug_info, current_info=None, save_path=None):
    # debug_info = copy.deepcopy(debug_info)
    l_pred_bbox = debug_info['pred_bbox']  # 5 x 4 list
    l_gt_bbox = debug_info['gt_bbox']  # N x (N, 1, 4)
    template = debug_info['template']  # (3, Ht, Wt)
    l_search = debug_info['search']  # N x (3, Hs, Ws)
    l_score_map = debug_info['pred_score_map']  # N x (H, W)
    l_pred_score = debug_info['pred_score']  # N x (1)
    if current_info is not None:
        l_pred_bbox.append(current_info['pred_bbox'])
        l_gt_bbox.append(None)
        l_search.append(current_info['search'])
        l_score_map.append(current_info['pred_score_map'])
        l_pred_score.append(current_info['pred_score'])
    out_size = (l_search[0].shape[1], l_search[0].shape[2])

    preprocessor = Preprocessor_wo_mask()

    template_image = preprocessor.inverse_process(template)
    pad = (l_search[0].shape[1] - template.shape[1]) // 2
    template_image = cv2.copyMakeBorder(template_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, (0, 0, 0))
    template_image = torch.tensor(template_image).permute(2, 0, 1)
    im_grid = [
        [template_image],
        [torch.zeros_like(template_image)]
    ]

    for search, pred_bbox, gt_bbox, score_map, pred_score in zip(l_search, l_pred_bbox, l_gt_bbox, l_score_map, l_pred_score):
        search_image = preprocessor.inverse_process(search)
        search_image_vis = draw_bbox(search_image.copy(), pred_bbox, (255, 0, 0))
        if gt_bbox is not None:
            gt_bbox = box_xywh_to_cxcywh(gt_bbox)[0]
            search_image_vis = draw_bbox(search_image_vis, gt_bbox, (0, 255, 0))
        search_image_vis = cv2.putText(search_image_vis, 'Stage1: %.2f' % float(score_map.max()), (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        search_image_vis = cv2.putText(search_image_vis, 'Stage2: %.2f' % float(pred_score), (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        im_grid[0].append(torch.tensor(search_image_vis).permute(2, 0, 1))

        score_map = vis_attn(score_map, out_size, search_image)
        im_grid[1].append(score_map)

    vis = torch.cat([torch.stack(x) for x in im_grid])
    vis = torchvision.utils.make_grid(vis, nrow=len(im_grid[0]))
    if save_path is None:
        imgcat(vis)
        print('')
        ipdb.set_trace()
    else:
        vis = vis.permute(1, 2, 0).numpy()[:, :, ::-1]
        cv2.imwrite(save_path, vis)


def linker_rnn_vis(debug_info, ps, vis_layers=None, attn_weights=None, save_path=None, gt_bbox=None):
    """

    Args:
        template: (3, H1, W1)
        prev_search: (3, H2, W2)
        search: (3, H2, W2)
        pred_bbox: (4)
        l_prev_search_patch_ids: (L, N)
        l_search_patch_ids: (L, N)
        vis_layers:
        attn_weights: list of L (h, N, N)
        save_path:

    Returns:
    """
    template = debug_info['template']
    search = debug_info['search']
    pred_bbox = debug_info['pred_bbox']
    score_map = debug_info['score_map']
    pred_score = debug_info['pred_score']
    l_s_patch_ids = debug_info['l_s_patch_ids']

    _, _, Wt = template.shape
    Wt //= 16
    Nt = Wt ** 2
    _, _, Ws = search.shape
    Ws //= 16
    Ns = Ws ** 2
    out_size = (search.shape[1], search.shape[2])
    preprocessor = Preprocessor_wo_mask()

    template_image = preprocessor.inverse_process(template)
    pad = (search.shape[1] - template.shape[1]) // 2
    template_image = cv2.copyMakeBorder(template_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, (0, 0, 0))

    search_image = preprocessor.inverse_process(search)
    search_overlay = get_patch_overlay(search_image, l_s_patch_ids[-1])
    search_image_eli = cv2.addWeighted(search_image, 0.3, search_overlay, 0.7, 0)
    search_image_eli = draw_bbox(search_image_eli, pred_bbox, (255, 0, 0))
    cv2.circle(search_image_eli, (int(search.shape[1] * 0.5), int(search.shape[1] * 0.5)), 3,
               (0, 0, 255), -1)
    if gt_bbox is not None:
        search_image_eli = draw_bbox(search_image_eli, gt_bbox, (0, 255, 0))

    if score_map is not None:
        max_score = score_map.max()
        score_map = vis_attn(score_map, out_size, search_image)
        score_map_im = score_map.permute(1, 2, 0).numpy()
        score_map_im = cv2.putText(score_map_im, 'Max: %.2f' % float(max_score), (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if pred_score is not None:
            pred_score = torch.sigmoid(pred_score)
            score_map_im = cv2.putText(score_map_im, 'Score: %.2f' % float(pred_score), (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        score_map = torch.tensor(score_map_im).permute(2, 0, 1)
    else:
        score_map = torch.zeros((3, *out_size))

    im_grid = [[
        torch.tensor(template_image).permute(2, 0, 1),
        torch.zeros((3, *out_size), dtype=torch.uint8) if ps is None else torch.tensor(ps).permute(2, 0, 1),
        torch.tensor(search_image_eli).permute(2, 0, 1),
        score_map
    ], [], []]

    # visualize attention map
    t_center = (Wt // 2, Wt // 2)  # center of template
    l_s_patch_ids.insert(0, torch.arange(Ns))

    if attn_weights[0] is None:
        im_grid = [im_grid[0]]
    else:
        for i, l in enumerate(vis_layers):
            # attn_weights contains multi-head attention of L layers
            # calculate the average of attention of all heads
            attn = attn_weights[i][0].mean(dim=0).squeeze().cpu()  # (Ns, Nt + Ns + Nt)

            # complete eliminated parts
            cur_Ns = l_s_patch_ids[l].shape[0]

            ## template to search attention
            t_s_attn = torch.zeros(Ns, Nt)
            t_s_attn[l_s_patch_ids[l], :] = attn[:, :Nt]
            t_s_attn = t_s_attn.reshape(Ws, Ws, Wt, Wt)[..., t_center[0], t_center[1]]  # (Ws, Ws)
            im_grid[1].append(vis_attn(t_s_attn, out_size, search_image))

            ## ps to search attention
            ps_s_attn = torch.zeros(Ns, Nt)
            if attn.shape[1] == 2 * Nt + cur_Ns:
                ps_s_attn[l_s_patch_ids[l], :] = attn[:, Nt+cur_Ns:]
                ps_s_attn = ps_s_attn.reshape(Ws, Ws, Wt, Wt)[..., t_center[0], t_center[1]]  # (Ws, Ws)
            im_grid[2].append(vis_attn(ps_s_attn, out_size, search_image))

    vis = torch.cat([torch.stack(x) for x in im_grid])
    vis = torchvision.utils.make_grid(vis, nrow=len(im_grid[0]))
    if save_path is None:
        imgcat(vis)
        print('')
        ipdb.set_trace()
    else:
        vis = vis.permute(1, 2, 0).numpy()[:, :, ::-1]
        cv2.imwrite(save_path, vis)