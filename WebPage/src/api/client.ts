// src/api/client.ts
// API client for communicating with the FastAPI backend

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

// ---- Types that match the backend/Swagger ----
export interface BackendMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface StartConversationRequest {
  conversation_id: string; // required by backend
}

export interface StartConversationResponse {
  conversation_id: string;
  question?: string;
  question_id?: string;
  done?: boolean;
  // backend might also include these sometimes
  context?: string;
  transcript?: BackendMessage[];
}

export interface ChatRequest {
  message: string;
}

// Backend might return different keys depending on implementation.
// We'll normalize in code.
export interface ChatResponseRaw {
  response?: string;
  answer?: string;
  message?: string;
  next_question?: string;
  question?: string;
  question_id?: string;
  done?: boolean;
  context?: string;
  transcript?: BackendMessage[];
}

export interface ChatResponseNormalized {
  text: string; // the assistant text to display
  next_question?: string;
  question_id?: string;
  done?: boolean;
  context?: string;
  transcript?: BackendMessage[];
}

export interface AnswerRequest {
  answer: string;
}

export interface AnswerResponse {
  next_question?: string;
  question?: string;
  question_id?: string;
  done?: boolean;
  context?: string;
  transcript?: BackendMessage[];
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
    console.log("🔧 API Client initialized with base URL:", this.baseUrl);
  }

  private async readError(response: Response): Promise<string> {
    const text = await response.text();
    try {
      const json = JSON.parse(text);
      if (json?.detail) {
        return Array.isArray(json.detail)
          ? json.detail.map((e: any) => e.msg).join(", ")
          : String(json.detail);
      }
      return JSON.stringify(json);
    } catch {
      return text || response.statusText;
    }
  }

  /**
   * Start a new conversation
   * Backend REQUIRES { conversation_id: string }
   */
  async startConversation(
    conversationId: string = `patient-${crypto.randomUUID?.() ?? Date.now()}`
  ): Promise<StartConversationResponse> {
    console.log("🚀 Starting new conversation...");
    const url = `${this.baseUrl}/api/v1/conversations/start`;
    console.log("📍 POST", url);

    const payload: StartConversationRequest = { conversation_id: conversationId };

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });

    console.log("📊 Response status:", response.status);

    if (!response.ok) {
      const msg = await this.readError(response);
      console.error("❌ Start conversation error response:", msg);
      throw new Error(`Failed to start conversation: ${msg}`);
    }

    const data = (await response.json()) as StartConversationResponse;
    console.log("✅ Conversation started:", data);
    return data;
  }

  /**
   * Send a chat message
   * We normalize the backend response into { text: string }
   */
  async sendChatMessage(
    conversationId: string,
    message: string
  ): Promise<ChatResponseNormalized> {
    console.log("💬 Sending chat message...");
    const url = `${this.baseUrl}/api/v1/conversations/${conversationId}/chat`;
    console.log("📍 POST", url);

    const payload: ChatRequest = { message };

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });

    console.log("📊 Response status:", response.status);

    if (!response.ok) {
      const msg = await this.readError(response);
      console.error("❌ Chat error response:", msg);
      throw new Error(`Failed to send message: ${msg}`);
    }

    const raw = (await response.json()) as ChatResponseRaw;
    console.log("✅ Chat raw response:", raw);

    const text =
      raw.response ??
      raw.answer ??
      raw.message ??
      raw.next_question ??
      raw.question ??
      "";

    return {
      text,
      next_question: raw.next_question ?? raw.question,
      question_id: raw.question_id,
      done: raw.done,
      context: raw.context,
      transcript: raw.transcript,
    };
  }

  /**
   * Submit an answer to a questionnaire question
   */
  async submitAnswer(
    conversationId: string,
    answer: string
  ): Promise<AnswerResponse> {
    console.log("📝 Submitting answer...");
    const url = `${this.baseUrl}/api/v1/conversations/${conversationId}/answer`;
    console.log("📍 POST", url);

    const payload: AnswerRequest = { answer };

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });

    console.log("📊 Response status:", response.status);

    if (!response.ok) {
      const msg = await this.readError(response);
      console.error("❌ Answer error response:", msg);
      throw new Error(`Failed to submit answer: ${msg}`);
    }

    const data = (await response.json()) as AnswerResponse;
    console.log("✅ Answer response:", data);
    return data;
  }

  async getConversationState(conversationId: string): Promise<any> {
    const url = `${this.baseUrl}/api/v1/conversations/${conversationId}/state`;
    console.log("📍 GET", url);

    const response = await fetch(url, {
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      const msg = await this.readError(response);
      throw new Error(`Failed to get state: ${msg}`);
    }

    return response.json();
  }

  async getConversationSummary(conversationId: string): Promise<any> {
    const url = `${this.baseUrl}/api/v1/conversations/${conversationId}/summary`;
    console.log("📍 GET", url);

    const response = await fetch(url, {
      headers: { Accept: "application/json" },
    });

    if (!response.ok) {
      const msg = await this.readError(response);
      throw new Error(`Failed to get summary: ${msg}`);
    }

    return response.json();
  }
}

export const apiClient = new ApiClient();
export default ApiClient;
