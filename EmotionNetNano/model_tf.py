import tensorflow as tf

from tensorflow.keras import layers , regularizers , Model


def emotion_nano_model(input_shape=(48,48,1) , num_classes=7 ):
  ##
  filters1=[11,9,11,8,11,7,11,27]
  filters2=[27,19,27,26,27,36]
  filters3=[64,39,64,24,64]
  names0=['1','2','3','4','5','6','7','8']
  names1=['9','10','11','12','13','14']
  names2=['15','16','17','18','19','20']
  ##

  #fundmental block
  inputs = tf.keras.Input(shape=input_shape)
  layer1 = layers.Conv2D(filters1[0], 3 , activation='relu', padding='same',name=names0[0])(inputs)
  layer2 = layers.Conv2D(filters1[1], 3 , activation='relu', padding='same',name=names0[1])(layer1)
  layer3 = layers.Conv2D(filters1[2], 3 , activation='relu', padding='same',name=names0[2])(layer2)
  layer4 = layers.Conv2D(filters1[3], 3 , activation='relu', padding='same',name=names0[3])(layer1+layer3)
  layer5 = layers.Conv2D(filters1[4], 3 , activation='relu', padding='same',name=names0[4])(layer4)
  layer6 = layers.Conv2D(filters1[5], 3 , activation='relu', padding='same',name=names0[5])(layer1+layer3+layer5)
  layer7 = layers.Conv2D(filters1[6], 3 , activation='relu', padding='same',name=names0[6])(layer6)
  layer8 = layers.Conv2D(filters1[7], 3 , activation='relu', padding='same', strides=(2,2),name=names0[7])(layer1+layer5+layer7)

  #1x1 conv layer 1
  identity1 = layers.Conv2D(27,1,strides=(2,2), name='identity1')(layer1+layer3+layer5)


  #cNN Block 1
  layer1_c1 = layers.Conv2D(filters2[0], 3 , activation='relu', padding='same',name=names1[0])(layer8)
  layer2_c1 = layers.Conv2D(filters2[1], 3 , activation='relu', padding='same',name=names1[1])(layer1_c1+identity1)
  layer3_c1 = layers.Conv2D(filters2[2], 3 , activation='relu', padding='same',name=names1[2])(layer2_c1)
  layer4_c1 = layers.Conv2D(filters2[3], 3 , activation='relu', padding='same',name=names1[3])(layer3_c1+layer1_c1)
  layer5_c1 = layers.Conv2D(filters2[4], 3 , activation='relu', padding='same',name=names1[4])(layer4_c1)
  layer6_c1 = layers.Conv2D(filters2[5], 3 , activation='relu', padding='same',strides=(2,2),name=names1[5])(layer3_c1+layer5_c1+layer1_c1+identity1)


  #1x1 conv layer 2
  identity2 = layers.Conv2D(64,1,strides=(2,2), name='identity2')(layer3_c1 +layer5_c1+identity1 +layer8)


  #CNN Block 2
  layer1_c2 = layers.Conv2D(filters3[0], 3 , activation='relu', padding='same',name=names2[0])(layer6_c1)
  layer2_c2 = layers.Conv2D(filters3[1], 3 , activation='relu', padding='same',name=names2[1])(layer1_c2+identity2)
  layer3_c2 = layers.Conv2D(filters3[2], 3 , activation='relu', padding='same',name=names2[2])(layer2_c2)
  layer4_c2 = layers.Conv2D(filters3[3], 3 , activation='relu', padding='same',name=names2[3])(layer3_c2+layer1_c2+identity2)
  layer5_c2 = layers.Conv2D(filters3[4], 3 , activation='relu', padding='same',name=names2[4])(layer4_c2)
  layer6_c2 = layers.AveragePooling2D((12,12),name=names2[5])(layer3_c2+layer5_c2+layer1_c2+identity2)


  #dense 
  dense = layers.Dense(num_classes)(layer6_c2)
  output= layers.Activation('softmax', name='softmax')(dense)
  model = Model(inputs= inputs , outputs= output)

  return model  

mymodel= emotion_nano_model()
mymodel.compile(
    optimizer= tf.keras.optimizers.Adam(lr=1e-3) ,
    loss= tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
mymodel.summary()


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode="nearest",
        validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
        "../data/train",
        target_size=(48, 48),
        batch_size=256,
        class_mode='sparse',
        subset='training',
        color_mode='grayscale')

validation_generator = train_datagen.flow_from_directory(
        "../data/test",
        target_size=(48, 48),
        batch_size=256,
        class_mode='sparse',
        subset='validation',
        color_mode='grayscale')

def scheduler(epoch, lr):
  lr0=1e-3
  if epoch >=81 and epoch <121:
    return lr0*1e-1
  elif epoch >=121 and epoch <161:
    return lr0*1e-2
  elif epoch >=161 and epoch <181:
    return lr0*1e-3
  elif epoch >=181:
    return lr0*0.5*1e-3
  else:
    return lr
  
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

checkpoint = tf.keras.callbacks.ModelCheckpoint('model{epoch:08d}.h5', period=20) 

mymodel.fit(train_generator,validation_data=validation_generator,epochs=200,batch_size=100,
            callbacks=[callback, checkpoint],verbose=2)