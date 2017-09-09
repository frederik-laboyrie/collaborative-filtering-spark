'''non-general framework for
   teacher student from larger 
   multi-resolution CNN to smaller
   multi-resolution TT_CNN
   working parameters currently
   x x 6 5
'''

from multires_CNN import *
from multires_TT_CNN import *
from data_preprocessing import load_standardized_multires
from keras.models import Model
from keras.layers import Activation
from keras.layers.merge import Add
import sys

tt_input_shape = [10, 26, 19, 6]
tt_output_shape = [10, 26, 19, 6]
tt_ranks = [1, 3, 3, 3, 1]

def get_data():
    multires_data, labels = load_standardized_multires()
    full = multires_data[0]
    med = multires_data[1]
    low = multires_data[2]
    return multires_data, labels, full, med, low

def train_teacher(multires_data, labels, full, med, low):
    teacher = multires_CNN(int(sys.argv[1]), int(sys.argv[2]), multires_data)
    teacher.fit([full, med, low], labels, epochs=10)
    return teacher

def get_student(tt_input_shape, tt_output_shape,
                tt_ranks, multires_data):
    student = multires_TT_CNN(int(sys.argv[3]), 
                              int(sys.argv[4]),
                              tt_input_shape, 
                              tt_output_shape,
                              tt_ranks, multires_data)
    return student

def make_teacher_untrainable(teacher):
    for i in range(len(teacher.layers)):
        setattr(teacher.layers[i], 'trainable', False)
    return teacher

def negativeActivation(x):
    return -x

def get_teacher_student_loss(teacher, student):
    negativeRight = Activation(negativeActivation)(student.output) 
    diff = Add()([teacher.output, negativeRight])
    return diff

def compile_full_model(teacher, student):
    model = Model(inputs=[teacher.input[0], teacher.input[1], teacher.input[2],
                          student.input[0], student.input[1], student.input[2]], 
                  outputs=[diff])

    model.compile(loss='mean_squared_error', 
                  optimizer='Adam', metrics=['acc'])
    return model

def main(tt_input_shape, tt_output_shape, tt_ranks):
    multires_data, labels, full, med, low = get_data()

    print('training teacher')
    teacher = train_teacher(multires_data, labels, 
                            full, med, low)

    print('compiling student')
    student = get_student(tt_input_shape, tt_output_shape,
                          tt_ranks, multires_data)

    static_teacher = make_teacher_untrainable(teacher)
    loss = get_teacher_student_loss(static_teacher, student)

    model = compile_full_model(static_teacher, student)
    model.summary(line_length=150)

    print('training student from teacher')
    model.fit([full, med, low, full, med, low], [labels],
               batch_size=1, nb_epoch=10)
    print(student.evaluate(full, med, low))

if __name__ == '__main__':
    main(tt_input_shape, tt_output_shape, tt_ranks)
