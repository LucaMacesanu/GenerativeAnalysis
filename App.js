import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

const App = () => {
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [storedUsers, setStoredUsers] = useState(
    JSON.parse(localStorage.getItem("users")) || {}
  );
  const [activeChat, setActiveChat] = useState(0);
  const [chats, setChats] = useState([]);
  const [input, setInput] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState(
    localStorage.getItem("user") ? true : false
  );
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (isAuthenticated) {
      const currentUser = localStorage.getItem("user");
      const storedChats = JSON.parse(localStorage.getItem(`chats_${currentUser}`)) || [
        {
          id: 0,
          title: "Chat 1",
          messages: [{ text: "Hello! How can I help you?", sender: "bot" }],
        },
      ];
      setChats(storedChats);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (isAuthenticated) {
      localStorage.setItem(`chats_${user}`, JSON.stringify(chats));
    }
  }, [chats, isAuthenticated, user]);

  const handleRegister = () => {
    if (user && password) {
      if (!storedUsers[user]) {
        const updatedUsers = { ...storedUsers, [user]: password };
        localStorage.setItem("users", JSON.stringify(updatedUsers));
        setStoredUsers(updatedUsers);
        localStorage.setItem("user", user);
        setIsAuthenticated(true);
      } else {
        alert("Username already exists!");
      }
    }
  };

  const handleLogin = () => {
    if (storedUsers[user] && storedUsers[user] === password) {
      localStorage.setItem("user", user);
      setIsAuthenticated(true);
    } else {
      alert("Invalid username or password");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("user");
    setUser("");
    setPassword("");
    setChats([]);
    setIsAuthenticated(false);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    setChats((prevChats) =>
      prevChats.map((chat) =>
        chat.id === activeChat
          ? {
              ...chat,
              messages: [...chat.messages, { text: input, sender: "user" }],
            }
          : chat
      )
    );

    try {
      const response = await axios.post("http://127.0.0.1:5000/query", { question: input });
      const botMessage = { text: response.data.response, sender: "bot" };
      setChats((prevChats) =>
        prevChats.map((chat) =>
          chat.id === activeChat
            ? {
                ...chat,
                messages: [...chat.messages, botMessage],
              }
            : chat
        )
      );
    } catch (error) {
      console.error("Error fetching response:", error);
    }

    setInput("");
  };

  return (
    <div className="chat-container">
      {isAuthenticated ? (
        <div className="chat-main">
          <button onClick={handleLogout} className="logout-btn">Logout</button>
          <div className="messages">
            {chats[activeChat]?.messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                <div className="bubble">{msg.text}</div>
              </div>
            ))}
          </div>
          <div className="input-area">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a message..."
            />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>
      ) : (
        <div className="login-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', textAlign: 'center' }}>
          <h2 style={{ fontSize: '32px', marginBottom: '20px' }}>TI Generative AI Tool</h2>
          <input type="text" placeholder="Username" value={user} onChange={(e) => setUser(e.target.value)} style={{ marginBottom: '10px', padding: '10px', fontSize: '16px' }} />
          <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} style={{ marginBottom: '20px', padding: '10px', fontSize: '16px' }} />
          <button onClick={handleLogin} style={{ marginBottom: '10px', padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}>Login</button>
          <button onClick={handleRegister} style={{ padding: '10px 20px', fontSize: '16px', cursor: 'pointer' }}>Register</button>
        </div>
      )}
    </div>
  );
};

export default App;
