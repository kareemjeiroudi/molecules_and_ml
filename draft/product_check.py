from itertools import product

first_names = ['Kareem', 'Sarah', 'Kristina', 'Leonardo']
middle_names = ['widdles', 'sizzles', 'Jesse', 'Jaquer']
last_names = ['Jeiroudi', 'Dussay', 'Preuer', 'Da Vinci']
the_list = [first_names, middle_names, last_names]

prod = product(first_names, middle_names, last_names)
f = open('myfile.txt', 'w')
for pro in prod:
	f.write(str(pro)+'\n')
f.close()