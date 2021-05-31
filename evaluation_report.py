import pandas as pd
import operator
import numpy as np
from tabulate import tabulate

import sys;sys.path.append('..')
from utils.abs_path import abs_path

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    # Assign variable names to coordinates for clarity
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = xi2 - xi1
    inter_height = yi2 - yi1
    inter_area = max(inter_width, 0) * max(inter_height, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou

def main():
    OUTPUT_FOLDER_EVAL = abs_path('../../data/eval_output')

    df_eval = pd.read_csv(OUTPUT_FOLDER_EVAL + '\\' + 'eval.csv',
                          names=['type_p', 'class','x', 'y', 'w', 'h', 'img_name', 'name_subclass', 'width', 'height',
                                 'distance', 'confidence'], index_col=False)

    df_eval.loc[(df_eval.type_p == 'hand'), 'x'] = round(df_eval.x / 100 * df_eval.width)
    df_eval.loc[(df_eval.type_p == 'hand'), 'y'] = round(df_eval.y / 100 * df_eval.height)
    df_eval.loc[(df_eval.type_p == 'hand'), 'w'] = round(df_eval.w / 100 * df_eval.width)
    df_eval.loc[(df_eval.type_p == 'hand'), 'h'] = round(df_eval.h / 100 * df_eval.height)

    df_eval['match'] = np.nan
    df_eval['iou'] = 0
    df_eval['is_math'] = False
    df_eval['is_iou'] = False

    images = df_eval.img_name.unique()

    for image in images:
        hand = df_eval.loc[(df_eval.img_name == image) & (df_eval.type_p == 'hand')]
        model = df_eval.loc[(df_eval.img_name == image) & (df_eval.type_p == 'model')]
        for index, row in hand.iterrows():
            box = (row.x, row.y, row.w, row.h)
            iou_value = [iou(box, (row1.x, row1.y, row1.w, row1.h)) for _, row1 in model.iterrows()]
            if iou_value:
                indexs = model.index.to_list()
                m_index, max_value = max(enumerate(iou_value), key=operator.itemgetter(1))
                max_index = indexs[m_index]
                if max_value:
                    df_eval.loc[index, ['match', 'iou']] = [max_index, max_value]
                    df_eval.loc[max_index, ['match', 'iou']] = [index, max_value]
                    df_eval.loc[index, 'is_iou'] = True
                    if row.name_subclass == df_eval.loc[max_index, 'name_subclass']:
                        df_eval.loc[index, 'is_math'] = True

    tab = df_eval.groupby(['type_p', 'class'])['name_subclass'].count()
    tab = tab.reset_index()

    print(tabulate(tab, headers=["Marking method", "# class", "Product count"],
                   tablefmt='psql', showindex=False))

    tab = df_eval.loc[df_eval.type_p == 'hand'].groupby(['is_iou', 'is_math'])['name_subclass'].count()
    tab = tab.reset_index()

    print(tabulate(tab, headers=["Marks matched", "Classified", "Product count"],
                   tablefmt='psql', showindex=False))

    print('% true object detection {}'.format(round(tab.loc[(tab.is_iou == True)].
                                                    name_subclass.sum() / sum(tab.name_subclass) * 100, 2)))

    print('% true object detection and classification {}'.format(
        round(tab.loc[(tab.is_iou == True) & (tab.is_math == True)].
              name_subclass.sum() / sum(tab.name_subclass) * 100, 2)))


if __name__ == '__main__':
    main()