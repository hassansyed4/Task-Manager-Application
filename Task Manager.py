import json
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

TASKS_FILE = 'tasks.json'


# Task class to handle task data
class Task:
    def __init__(self, title, description, due_date, priority, completed=False):
        self.title = title
        self.description = description
        self.due_date = due_date
        self.priority = priority
        self.completed = completed

    def __str__(self):
        status = "Completed" if self.completed else "Pending"
        return f"Title: {self.title}, Due: {self.due_date}, Priority: {self.priority}, Status: {status}"

#3
def load_tasks():
    try:
        with open(TASKS_FILE, 'r') as file:
            data = json.load(file)
            return [Task(task['title'], task['description'], task['due_date'], task['priority'],
                         task.get('completed', False)) for task in data]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_tasks(tasks):
    with open(TASKS_FILE, 'w') as file:
        json.dump([task.__dict__ for task in tasks], file)


def add_task(tasks):
    title = input("Title: ")
    description = input("Description: ")
    due_date = input("Due Date (YYYY-MM-DD): ")

    # Use model to predict priority based on task description
    priority = predict_priority(description)
    print(f"Predicted Priority: {priority}")

    tasks.append(Task(title, description, due_date, priority))
    save_tasks(tasks)


def view_tasks(tasks):
    if not tasks:
        print("No tasks available.")
    for idx, task in enumerate(tasks, 1):
        print(f"{idx}. {task}")


def mark_completed(tasks):
    view_tasks(tasks)
    try:
        task_index = int(input("Enter task number to mark as completed: ")) - 1
        if 0 <= task_index < len(tasks):
            tasks[task_index].completed = True
            save_tasks(tasks)
            print(f"Task '{tasks[task_index].title}' marked as completed.")
        else:
            print("Invalid task number.")
    except ValueError:
        print("Please enter a valid number.")


# Machine Learning Model for Task Priority Prediction
def train_priority_model():
    # Sample training data
    descriptions = [
        "Complete the project report", "Prepare for meeting with the client",
        "Buy groceries", "Attend the doctor's appointment",
        "Fix the bug in the system", "Complete the code review",
        "Write the technical documentation", "Review the marketing strategy"
    ]
    priorities = ["High", "High", "Low", "Medium", "High", "Medium", "Low", "Medium"]

    # Create a text classification pipeline
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(descriptions, priorities)

    return model


def predict_priority(description):
    # Predict task priority using the trained model
    model = train_priority_model()
    return model.predict([description])[0]


# Main program flow   # 2
def main():
    tasks = load_tasks()
    while True:
        print("\nTask Manager")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Mark Task as Completed")
        print("4. Exit")
        choice = input("Choose an option: ")
        if choice == '1':
            add_task(tasks)
        elif choice == '2':
            view_tasks(tasks)
        elif choice == '3':
            mark_completed(tasks)
        elif choice == '4':
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__": #1
    main()
