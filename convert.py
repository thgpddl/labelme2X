import argparse


from dataset_conversion import *

labelme2voc = LabelMe2VOC().convert
labelme2coco = LabelMe2COCO().convert


def dataset_conversion(source, to, pics, anns, save_dir):
    if source.lower() == 'labelme' and to.lower() == 'pascalvoc':
        labelme2voc(pics, anns, save_dir)
    elif source.lower() == 'labelme' and to.lower() == 'mscoco':
        labelme2coco(pics, anns, save_dir)
    else:
        raise Exception("Converting from {} to {} is not supported.".format(
            source, to))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default='labelme', help="原始标注类型，默认为labelme")
    parser.add_argument('--to', help="表示数据需要转换成为的格式，支持'pascalvoc','mscoco'")
    parser.add_argument('--pics', help="指定原图所在的目录路径")
    parser.add_argument('--anns', help="指定标注文件所在的目录路径")
    parser.add_argument('--save_dir ', help="指定保存路径")
    args = parser.parse_args()

    dataset_conversion(args.source, args.to, args.pics, args.anns, args.save_dir)


