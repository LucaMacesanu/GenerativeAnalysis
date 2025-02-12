import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const WEBSOCKET_URL = "wss://pdobdpl064.execute-api.us-east-1.amazonaws.com/production/";

const App = () => {
  const [messages, setMessages] = useState([{ text: "Hello! How can I help you?", sender: "bot" }]);
  const [input, setInput] = useState("");
  const [socket, setSocket] = useState(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(WEBSOCKET_URL);

    ws.onopen = () => console.log("Connected to WebSocket API");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessages((prev) => [...prev, { text: data.message, sender: "bot" }]);
    };
    ws.onclose = () => console.log("WebSocket Disconnected");
    ws.onerror = (error) => console.error("WebSocket Error:", error);

    setSocket(ws);

    return () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim() || !socket) return;
    const userMessage = { text: input, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);

    socket.send(JSON.stringify({ action: "sendMessage", message: input }));
    setInput("");
  };

  return (
      <div className="chat-container">
        <div className="chat-box">
          {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                {msg.sender === "user" ? (
                    <div className="message-bubble user-message">{msg.text}</div>
                ) : (
                    <div className="bot-message">{msg.text}</div>
                )}
              </div>
          ))}
          <div ref={chatEndRef} />
        </div>
        <div className="input-container">
          <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="chat-input"
              placeholder="Type a message..."
          />
          <button onClick={sendMessage} className="send-button">Send</button>
        </div>
      </div>
  );
};

export default App;