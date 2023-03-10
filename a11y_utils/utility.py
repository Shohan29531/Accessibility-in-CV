def read_lines(file_name):
    file_object = open(file_name, 'r')
    lines = file_object.readlines()

    return lines


def make_a11y_questions():
    lines = read_lines('../a11y_objects_of_interest.txt')
    questions = []

    for item in lines:
        question_str = "Is there a/an " + item.strip() + " in the scene?\n"
        questions.append( question_str )

    file_object = open('../a11y_questions_of_interest.txt', 'w')
    file_object.writelines( questions )
    file_object.close()



def get_a11y_questions( a11y_filename ):
    lines = read_lines( a11y_filename )

    questions = []

    for line in lines:
        questions.append( line.strip() )

    return questions    


# make_a11y_questions()
# print(get_a11y_questions())