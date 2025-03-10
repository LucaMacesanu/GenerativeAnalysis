require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

const app = express();
app.use(express.json());
app.use(cors());

const PORT = 3001;
const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/chatApp";
const JWT_SECRET = process.env.JWT_SECRET || "your_secret_key";

// Connect to MongoDB
mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
}).then(() => console.log("MongoDB Connected"))
  .catch(err => console.log("MongoDB Connection Error:", err));

// Define User Schema
const userSchema = new mongoose.Schema({
  username: { type: String, unique: true },
  password: String,
});

// Define Chat Schema
const chatSchema = new mongoose.Schema({
  username: String,
  chats: Array,
});

const User = mongoose.model("User", userSchema);
const Chat = mongoose.model("Chat", chatSchema);

// Register Route
app.post("/register", async (req, res) => {
  const { username, password } = req.body;

  const existingUser = await User.findOne({ username });
  if (existingUser) return res.status(400).json({ error: "User already exists" });

  const hashedPassword = await bcrypt.hash(password, 10);
  const newUser = new User({ username, password: hashedPassword });

  await newUser.save();
  res.json({ message: "User registered successfully" });
});

// Login Route
app.post("/login", async (req, res) => {
  const { username, password } = req.body;

  const user = await User.findOne({ username });
  if (!user) return res.status(400).json({ error: "Invalid credentials" });

  const isMatch = await bcrypt.compare(password, user.password);
  if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });

  const token = jwt.sign({ username }, JWT_SECRET, { expiresIn: "1h" });
  res.json({ message: "Login successful", token });
});

// Save Chat History
app.post("/save-chat", async (req, res) => {
  const { username, chats } = req.body;

  await Chat.findOneAndUpdate(
    { username },
    { username, chats },
    { upsert: true }
  );

  res.json({ message: "Chat history saved" });
});

// Retrieve Chat History
app.get("/get-chat/:username", async (req, res) => {
  const { username } = req.params;
  const chatData = await Chat.findOne({ username });

  res.json(chatData ? chatData.chats : []);
});

// Start Server
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
