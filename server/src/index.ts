import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

app.get('/api/health', (req, res) => {
  res.json({ message: 'Server is running!' });
});

app.get('/api/modules', (req, res) => {
  res.json({ modules: [] });
});

app.post('/api/modules', (req, res) => {
  const { name, content } = req.body;
  res.json({ message: 'Module saved successfully', name });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});