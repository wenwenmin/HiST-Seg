import os
import cv2
import numpy as np


# ============================================================
# 路径设置
# ============================================================
INPUT_PATH = r"D:\Project_2\Pixel_Seg\data\DuCIS\resized_data\label_resized_red.png"
OUTPUT_PATH = r"D:\Project_2\Pixel_Seg\data\DuCIS\resized_data\label_resized_corrected.png"

# 可选：保存检测到的红色区域，便于检查
DEBUG_RED_MASK_PATH = r"D:\Project_2\Pixel_Seg\data\DuCIS\resized_data\debug_red_mask.png"
DEBUG_FILLED_MASK_PATH = r"D:\Project_2\Pixel_Seg\data\DuCIS\resized_data\debug_red_filled_regions.png"


def main():
    # ========================================================
    # 1. 读取带红色标记的标签图
    # ========================================================
    image = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"无法读取图像：{INPUT_PATH}")

    print(f"Input image shape: {image.shape}")

    # OpenCV读取格式为 BGR
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ========================================================
    # 2. 提取红色像素
    #
    # HSV中的红色跨越0度，因此需要两个范围
    # ========================================================
    lower_red_1 = np.array([0, 80, 80], dtype=np.uint8)
    upper_red_1 = np.array([15, 255, 255], dtype=np.uint8)

    lower_red_2 = np.array([165, 80, 80], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

    red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    # ========================================================
    # 3. 对红色线条做轻微闭运算
    #
    # 防止人工画圈存在很小的断裂，使轮廓能够闭合
    # ========================================================
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    red_mask_closed = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # 再轻微膨胀一下，让轮廓更加连续
    red_mask_closed = cv2.dilate(
        red_mask_closed,
        np.ones((3, 3), np.uint8),
        iterations=1
    )

    # ========================================================
    # 4. 查找红色封闭轮廓
    # ========================================================
    contours, _ = cv2.findContours(
        red_mask_closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"检测到红色轮廓数量: {len(contours)}")

    # ========================================================
    # 5. 创建需要删除的区域mask
    #
    # 红色圈本身 + 红色圈内部全部设为255
    # ========================================================
    remove_mask = np.zeros(
        image.shape[:2],
        dtype=np.uint8
    )

    valid_contours = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        print(f"Contour {i}: area = {area:.2f}")

        # 过滤极小的红色噪声
        # 如果你的圈很小，可以适当减小这个值
        if area < 20:
            continue

        valid_contours.append(contour)

        # thickness=-1 表示把轮廓内部全部填满
        cv2.drawContours(
            remove_mask,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=-1
        )

    print(f"有效红色封闭区域数量: {len(valid_contours)}")

    # ========================================================
    # 6. 确保原始红色线本身也全部删除
    # ========================================================
    remove_mask = cv2.bitwise_or(
        remove_mask,
        red_mask
    )

    # ========================================================
    # 7. 生成最终二值标签
    #
    # 原始标签：
    # 白色 = 癌症区域
    # 黑色 = 非癌症区域
    #
    # 先转换成灰度，再二值化
    # ========================================================
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 原始白色癌症区域设为255，其余设为0
    _, corrected_label = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # ========================================================
    # 8. 将红圈及红圈内部全部设为黑色
    # ========================================================
    corrected_label[remove_mask > 0] = 0

    # 最后再次确保输出严格为0/255二值图
    corrected_label = np.where(
        corrected_label > 127,
        255,
        0
    ).astype(np.uint8)

    # ========================================================
    # 9. 保存
    # ========================================================
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    cv2.imwrite(
        OUTPUT_PATH,
        corrected_label
    )

    # 保存调试图
    cv2.imwrite(
        DEBUG_RED_MASK_PATH,
        red_mask
    )

    cv2.imwrite(
        DEBUG_FILLED_MASK_PATH,
        remove_mask
    )

    print("\n处理完成！")
    print(f"修正后的标签保存到：")
    print(OUTPUT_PATH)

    print("\n调试文件：")
    print(f"红色像素检测：{DEBUG_RED_MASK_PATH}")
    print(f"红圈填充区域：{DEBUG_FILLED_MASK_PATH}")


if __name__ == "__main__":
    main()