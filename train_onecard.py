import argparse
from pathlib import Path
import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD, Adam
from generator import FaceGenerator, ValGenerator
from model import get_model, age_mae
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import pandas
from keras.applications.imagenet_utils import decode_predictions

#from efficientnet import EfficientNetB0
#from efficientnet import center_crop_and_resize, preprocess_input



def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--appa_dir", type=str, required=True,default="./data",
                        help="path to the APPA-REAL dataset")
    parser.add_argument("--utk_dir", type=str, default=None,
                        help="path to the UTK face dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=5000,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--opt", type=str, default="adam",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--count", type=str, default="0",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--freeze", type=str, default="0",
                        help="optimizer name; 'sgd' or 'adam'")
    parser.add_argument("--model_name", type=str, default="MobileNetV2",
                        help="model name: 'ResNet50' or 'InceptionResNetV2' or 'MobileNetV3_l' or 'MobileNetV3_s' or 'Efficientnet'")
    args = parser.parse_args()
    return args








class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008


def get_optimizer(opt_name, lr):
    if opt_name == "sgd":
        return SGD(lr=lr, momentum=0.9, nesterov=True)
    elif opt_name == "adam":
        return Adam(lr=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def main():
    args = get_args()
    appa_dir = args.appa_dir
    utk_dir = args.utk_dir
    model_name = args.model_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    freeze = args.freeze
    opt_name = args.opt
    count= args.count
    print("model_name :",model_name)
    print(model_name=="Efficientnet")

    if model_name == "ResNet50":
        image_size = 224
        
    if model_name == "InceptionResNetV2":
        image_size = 299
    if model_name =="MobileNetV2":
        image_size=224
        print("image_size :",image_size)
        
    if model_name =="MobileNetV3_l" or "MobileNetV3_s":
        image_size =224
    if model_name =="Efficientnet":
       image_size =300
        


    print("image_size :",image_size)
    train_gen = FaceGenerator(appa_dir, utk_dir=utk_dir, batch_size=batch_size, image_size=image_size)
    val_gen = ValGenerator(appa_dir, batch_size=1, image_size=image_size)
    if model_name == "ResNet50" or model_name == "InceptionResNetV2" or model_name =="MobileNetV2" or model_name=="Efficientnet" :
        model = get_model(model_name=model_name)
    elif model_name ==  "MobileNetV3_l":
        from mobilev3.mobilenet_v3_large import MobileNetV3_Large
        model = MobileNetV3_Large((224,224,3), 8).build()
    elif model_name ==  "MobileNetV3_s":
        from mobilev3.mobilenet_v3_small import MobileNetV3_Small
        model = MobileNetV3_Small((224,224,3), 8).build()
   

    opt = get_optimizer(opt_name, lr)
    for layer in model.layers:
        layer.trainable = True
    for i in range(int(freeze)):
        model.layers[i].trainable = False
    
    #print('\n how many layers :',len(model.layers))
    #for layer in model.layers:
  #      print(layer.name, ' is trainable? ', layer.trainable)


    
    model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])
    model.summary()
    output_dir = Path(__file__).resolve().parent.joinpath(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [LearningRateScheduler(schedule=Schedule(nb_epochs, initial_lr=lr)),
                 ModelCheckpoint(str(output_dir)+ "/weights.0917_asia_allage_megaage_"+str(freeze)+"_"+model_name+"_{epoch:03d}-{val_loss:.3f}.hdf5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="min")
                 ]

    hist = model.fit_generator(generator=train_gen,
                               epochs=nb_epochs,
                               validation_data=val_gen,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_dir.joinpath("history.npz")), history=hist.history)
    
    # Y_pred = model.predict_generator(val_gen)
    # #print("Y:",Y_pred)
    # y_pred = np.argmax(Y_pred, axis=1)
    # #print("y:",y_pred)
    # print('Confusion Matrix')
    # print(val_gen.classes())
    # print(type(val_gen.classes()))
    # print(confusion_matrix(val_gen.classes(), y_pred))
    # report_conf = confusion_matrix(val_gen.classes(), y_pred)
    # df_conf = pandas.DataFrame(report_conf)
    # df_conf.to_csv("./result_csv/eff_result/sel0719"+'conf_matrix_adam_'+count+"_"+model_name+'.csv',  sep=',')
    # print('Classification Report')
    # target_names = ['2', '2pn', '3', '4','5', 'blastocyst','earlybc','morula']
    # print(classification_report(val_gen.classes(), y_pred, target_names=target_names))
    # report = classification_report(val_gen.classes(), y_pred, target_names=target_names,output_dict=True)
    # df = pandas.DataFrame(report).transpose()
    # df.to_csv("./result_csv/eff_result/sel0719"+'_adam_'+count+"_"+model_name+'.csv',  sep=',')


if __name__ == '__main__':
    main()
