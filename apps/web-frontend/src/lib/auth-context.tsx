"use client";

import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

import type { UserResponse } from "@/lib/api";

type AuthState = {
  user: UserResponse | null;
  accessToken: string | null;
};

type AuthContextValue = {
  user: UserResponse | null;
  token: string | null;
  isLoading: boolean;
  setAuth: (user: UserResponse, accessToken: string) => void;
  logout: () => void;
};

const AUTH_STORAGE_KEY = "ticket-forge.auth";

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function readStoredAuth(): AuthState {
  if (typeof window === "undefined") {
    return { user: null, accessToken: null };
  }

  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) {
      return { user: null, accessToken: null };
    }

    const parsed = JSON.parse(raw) as AuthState;
    return {
      user: parsed.user ?? null,
      accessToken: parsed.accessToken ?? null,
    };
  } catch {
    return { user: null, accessToken: null };
  }
}

function persistAuth(next: AuthState) {
  if (typeof window === "undefined") {
    return;
  }

  if (!next.user || !next.accessToken) {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return;
  }

  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(next));
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuthState] = useState<AuthState>(() => readStoredAuth());

  function setAuth(user: UserResponse, accessToken: string) {
    const next = { user, accessToken };
    setAuthState(next);
    persistAuth(next);
  }

  function logout() {
    const next = { user: null, accessToken: null };
    setAuthState(next);
    persistAuth(next);
  }

  return (
    <AuthContext.Provider
      value={{
        user: auth.user,
        token: auth.accessToken,
        isLoading: false,
        setAuth,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
