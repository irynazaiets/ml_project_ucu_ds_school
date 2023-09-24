class Node:
    def __init__(self, data, next_node=None):
        self.data = data
        self.next_node = next_node
        
class MyStack:
    def __init__(self):
        self.head = None
        self.size = 0

    def __str__(self):
        current_node = self.head
        output = ""
        while current_node:
            output += str(current_node.data) + "->"
            current_node = current_node.next_node
        return output[:-2]

    def is_empty(self):
        return self.size == 0

    def push(self, value):
        new_node = Node(value)
        new_node.next_node = self.head
        self.head = new_node
        self.size += 1

    def pop(self):
        if self.is_empty():
            raise Exception("You are trying to pop an empty stack!")

        if self.size == 1:
            popped_element = self.head.data
            self.head = None
            self.size = 0
        else:
            current_node = self.head
            while current_node.next_node.next_node:
                current_node = current_node.next_node
            popped_element = current_node.next_node.data
            current_node.next_node = None
            self.size -= 1

        return popped_element

# Create a MyStack instance and test it
my_stack = MyStack()
my_stack.push(7)
my_stack.push(8)
my_stack.push(9)
my_stack.push(10)
my_stack.push(11)
my_stack.push(12)
my_stack.push(13)

print(my_stack)

my_stack.pop()  # Remove the last element

print(my_stack)
