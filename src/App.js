// App.js
import React, { useEffect, useRef, useState } from "react";
import './App.css';

const App = () => {
  const [activeChat, setActiveChat] = useState(0);
  const [chats, setChats] = useState([
    {
      id: 0,
      title: "Chat 1",
      messages: [{ text: "Hello! How can I help you?", sender: "bot" }],
    },
  ]);
  const [input, setInput] = useState("");
  const [socket, setSocket] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef(null);

  // Establish WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(
      "wss://pdobdpl064.execute-api.us-east-1.amazonaws.com/production/"
    );

    ws.onopen = () => console.log("Connected to WebSocket API");
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setIsTyping(false);
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

    return () => ws.close();
  }, [activeChat]);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats]);

  const createNewChat = () => {
    const newChatId = chats.length;
    setChats([
      ...chats,
      {
        id: newChatId,
        title: `Chat ${newChatId + 1}`,
        messages: [{ text: "Hello! How can I help you?", sender: "bot" }],
      },
    ]);
    setActiveChat(newChatId);
  };

  const sendMessage = () => {
    if (!input.trim() || !socket) return;

    // Add user's message locally
    setChats((prevChats) =>
      prevChats.map((chat) =>
        chat.id === activeChat
          ? {
              ...chat,
              messages: [...chat.messages, { text: input, sender: "user" }],
              title: chat.title.startsWith("New Chat")
                ? input.slice(0, 30) + (input.length > 30 ? "..." : "")
                : chat.title,
            }
          : chat
      )
    );

    setIsTyping(true);
    socket.send(JSON.stringify({ action: "sendMessage", message: input }));
    setInput("");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const currentChat = chats.find((chat) => chat.id === activeChat);

  return (
    <div className="chat-container">
      {/* Sidebar */}
      <div className="sidebar">
        <h2>Chats</h2>
        <button onClick={createNewChat} className="new-chat-btn">
          New Chat
        </button>
        <div className="chat-list">
          {chats.map((chat) => (
            <button
              key={chat.id}
              onClick={() => setActiveChat(chat.id)}
              className={`chat-item ${activeChat === chat.id ? "active" : ""}`}
            >
              {chat.title}
            </button>
          ))}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-main">
        {/* Messages */}
        <div className="messages">
          {currentChat?.messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              <div className="bubble">{msg.text}</div>
            </div>
          ))}

          {isTyping && (
            <div className="typing-indicator">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-area">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type a message..."
            rows="1"
          />
          <button onClick={sendMessage} className="send-btn" disabled={!input.trim()}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
