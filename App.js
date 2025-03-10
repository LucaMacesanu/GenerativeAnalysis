import React, { useEffect, useRef, useState } from "react";
import "./App.css";

const WEBSOCKET_URL = "wss://pdobdpl064.execute-api.us-east-1.amazonaws.com/production/";

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
  const [socket, setSocket] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (isAuthenticated) {
      const currentUser = localStorage.getItem("user");
      const storedChats =
        JSON.parse(localStorage.getItem(`chats_${currentUser}`)) || [
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

  useEffect(() => {
    if (isAuthenticated) {
      const ws = new WebSocket(WEBSOCKET_URL);

      ws.onopen = () => console.log("Connected to WebSocket API");
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setChats((prevChats) =>
          prevChats.map((chat) =>
            chat.id === activeChat
              ? {
                  ...chat,
                  messages: [...chat.messages, { text: data.message, sender: "bot" }],
                }
              : chat
          )
        );
      };
      ws.onclose = () => console.log("WebSocket Disconnected");
      ws.onerror = (error) => console.error("WebSocket Error:", error);

      setSocket(ws);

      return () => {
        ws.close();
      };
    }
  }, [isAuthenticated, activeChat]);

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

  const sendMessage = () => {
    if (!input.trim() || !socket) return;

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

    socket.send(JSON.stringify({ action: "sendMessage", message: input }));

    setInput("");
  };

  return (
    <div className="chat-container">
      {isAuthenticated ? (
        <div className="chat-main">
          <button onClick={handleLogout} className="logout-btn">
            Logout
          </button>
          <div className="messages">
            {chats[activeChat]?.messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                <div className="bubble">{msg.text}</div>
              </div>
            ))}
            <div ref={chatEndRef} />
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
        <div
          className="login-container"
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100vh",
            textAlign: "center",
          }}
        >
          <h2 style={{ fontSize: "32px", marginBottom: "20px" }}>TI Generative AI Tool</h2>
          <input
            type="text"
            placeholder="Username"
            value={user}
            onChange={(e) => setUser(e.target.value)}
            style={{ marginBottom: "10px", padding: "10px", fontSize: "16px" }}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ marginBottom: "20px", padding: "10px", fontSize: "16px" }}
          />
          <button
            onClick={handleLogin}
            style={{ marginBottom: "10px", padding: "10px 20px", fontSize: "16px", cursor: "pointer" }}
          >
            Login
          </button>
          <button
            onClick={handleRegister}
            style={{ padding: "10px 20px", fontSize: "16px", cursor: "pointer" }}
          >
            Register
          </button>
        </div>
      )}
    </div>
  );
};

export default App;
