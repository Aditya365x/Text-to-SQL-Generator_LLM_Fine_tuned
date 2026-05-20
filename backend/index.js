const express = require('express');
const cors = require('cors');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory storage
let todos = [
  { id: uuidv4(), text: 'Learn Node.js', completed: false },
  { id: uuidv4(), text: 'Build a To-Do App', completed: false },
  { id: uuidv4(), text: 'Master Express.js', completed: false },
];

// Routes

// GET all todos
app.get('/api/todos', (req, res) => {
  res.json(todos);
});

// GET a single todo by ID
app.get('/api/todos/:id', (req, res) => {
  const todo = todos.find(t => t.id === req.params.id);
  if (!todo) {
    return res.status(404).json({ message: 'Todo not found' });
  }
  res.json(todo);
});

// POST create a new todo
app.post('/api/todos', (req, res) => {
  const { text } = req.body;
  if (!text || text.trim() === '') {
    return res.status(400).json({ message: 'Text is required' });
  }

  const newTodo = {
    id: uuidv4(),
    text: text.trim(),
    completed: false,
  };

  todos.push(newTodo);
  res.status(201).json(newTodo);
});

// PUT update a todo
app.put('/api/todos/:id', (req, res) => {
  const { text, completed } = req.body;
  const todo = todos.find(t => t.id === req.params.id);

  if (!todo) {
    return res.status(404).json({ message: 'Todo not found' });
  }

  if (text !== undefined) {
    if (text.trim() === '') {
      return res.status(400).json({ message: 'Text cannot be empty' });
    }
    todo.text = text.trim();
  }

  if (completed !== undefined) {
    todo.completed = completed;
  }

  res.json(todo);
});

// DELETE a todo
app.delete('/api/todos/:id', (req, res) => {
  const index = todos.findIndex(t => t.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ message: 'Todo not found' });
  }

  todos.splice(index, 1);
  res.status(204).send();
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
