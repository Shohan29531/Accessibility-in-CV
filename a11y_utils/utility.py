def read_lines(file_name):
    file_object = open(file_name, 'r')
    lines = file_object.readlines()

    return lines


def make_a11y_questions():
    lines = read_lines('../a11y_objects_of_interest.txt')
    questiions = []

    for item in lines:
        questiion_str = "Is there a/an " + item.strip() + " in the scene?\n"
        questiions.append( questiion_str )

    file_object = open('../a11y_questions_of_interest.txt', 'w')
    file_object.writelines( questiions )
    file_object.close()



def get_a11y_questions():
    return  read_lines( '../a11y_questions_of_interest.txt' )


# make_a11y_questions()
# print(get_a11y_questions())