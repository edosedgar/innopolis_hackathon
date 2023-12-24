import sys
import numpy as np
import pandas as pd


def parse(string):
    if type(string) == type("string"):
        string = string.strip("[").strip("]")
        string = string.split(", ")
        answer = []
        for x in string:
            answer.append(float(x.strip("[").strip("]").strip(" ")))
        return answer
    else:
        return string


def calculate_iou(boxA, boxB):
    xA, yA, wA, hA = boxA
    xB, yB, wB, hB = boxB

    Ax1, Ay1, Ax2, Ay2 = xA - wA / 2, yA - hA / 2, xA + wA / 2, yA + hA / 2
    Bx1, By1, Bx2, By2 = xB - wB / 2, yB - hB / 2, xB + wB / 2, yB + hB / 2

    xA = max(Ax1, Bx1)
    yA = max(Ay1, By1)
    xB = min(Ax2, Bx2)
    yB = min(Ay2, By2)

    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    boxA_area = wA * hA
    boxB_area = wB * hB

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou


def fill_rbbox(x):
    if str(x) == "nan":
        return [[0,0,0,0]]
    else:
        return x

def parse_bbox(string):
    string = str(string).strip("[").strip("]").split(", ")
    counter = 0
    answer = []
    for x in string:
        if counter % 4 != 0:
            answer[-1].append(float(x.strip("[").strip("]")))
        else:
            answer.append([])
            answer[-1].append(float(x.strip("[").strip("]")))
        counter+=1
    return answer

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    This metrics is used as integral metric for mAP and IoU.
    solution = pd.DataFrame({"file_name":["id1_1","id2_1"],"bbox":[[[0,0,1,1]],[[0,0,1,1]]]})
    submission = pd.DataFrame({"file_name":["id1_1","id1_2"],"x":[0,1],"y":[1,2],"w":[0.5,0.5], "h":[0.5,0.5], "probab":[0.5,0.5]})
    print(df1)
    print(df2)
    print(score(df1,df2,"index"))
    '''
    submission = submission.set_index(row_id_column_name)
    submission["rbbox"] = submission["rbbox"].apply(parse_bbox)
    submission["probability"] = submission["probability"].apply(parse_bbox)
    submission["probability"] = submission["probability"].apply(lambda x: x[0])
    solution = solution.set_index(row_id_column_name)
    solution["bbox"] = solution["bbox"].apply(parse_bbox)

    total_gt_boxes = solution['bbox'].apply(lambda x: len(x)).sum()

    h = 0.01
    all_precisions = []
    all_recalls = []

    for i in range(1,100):
        local_submission = submission[submission["probability"].apply(lambda x: any(float(val) > h * i for val in x))].copy()
        del local_submission["probability"]
        df = solution.join(local_submission, how="outer")
        df["rbbox"] = df["rbbox"].apply(fill_rbbox)

        iou_list = []
        for index, row in df.iterrows():
            predicted_bboxes = row['rbbox']
            real_bboxes = row['bbox']


            iou_list_per_row = []
            for pred_box in predicted_bboxes:
                ious = [calculate_iou(pred_box, real_box) for real_box in real_bboxes]
                max_iou = max(ious) if ious else 0
                iou_list_per_row.append(max_iou)
            iou_list.append(iou_list_per_row)

        df['iou_list'] = iou_list

        df['true_positives'] = df['iou_list'].apply(lambda iou_list: sum(iou > 0.5 for iou in iou_list))
        true_positives = df["true_positives"].sum()
        all_positives = local_submission['rbbox'].apply(lambda x: len(x)).sum()

        if total_gt_boxes == 0 and all_positives == 0:
            precision = 1
            recall = 1
        elif total_gt_boxes == 0 and all_positives > 0:
            recall = 0
            precision = 0
        else:
            precision = true_positives / all_positives if all_positives > 0 else 0
            recall = true_positives / total_gt_boxes if total_gt_boxes > 0 else 0

        all_precisions.append(precision)
        all_recalls.append(recall)

    mAP = compute_ap(all_recalls, all_precisions)
    return mAP


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] and sys.argv[2]:
        solution = pd.read_csv(sys.argv[1])
        submission = pd.read_csv(sys.argv[2])
    else:
        print("No submission specified")
        sys.exit(1)
    print(score(solution, submission, 'file_name'))
