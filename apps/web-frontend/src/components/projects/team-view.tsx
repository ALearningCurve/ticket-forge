"use client";

import { useEffect, useState } from "react";
import { Loader2, Sparkles, ArrowRight, User } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";
import {
  getEngineerTicketRecommendations,
  type EngineerTicketRecommendationsResponse,
  type ProjectMember,
  type TicketResponse,
} from "@/lib/api";

interface TeamViewProps {
  members: ProjectMember[];
  tickets: TicketResponse[];
  projectSlug: string;
  sizePointsMap: { S: number; M: number; L: number; XL: number };
  weeklyPointsPerMember: number;
}

// Replace these with your own hex codes if you have a specific palette!
const AVATAR_COLORS = [
  "#6366f1",
  "#8b5cf6",
  "#06b6d4",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#ec4899",
  "#14b8a6",
];

function getAssignedTickets(
  tickets: TicketResponse[],
  userId: string
): TicketResponse[] {
  return tickets.filter((t) => t.assignee?.id === userId);
}

function getActiveTickets(
  tickets: TicketResponse[],
  userId: string
): TicketResponse[] {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  return tickets.filter((t) => {
    if (t.assignee?.id !== userId) return false;
    if (!t.due_date) return true;
    const due = new Date(t.due_date + "T00:00:00");
    return due >= today;
  });
}

function getAvailabilityLabel(
  tickets: TicketResponse[],
  userId: string
): { label: string; variant: "default" | "destructive" | "secondary"; colorClass: string } {
  const active = getActiveTickets(tickets, userId);
  if (active.length === 0) return { label: "Available", variant: "default", colorClass: "bg-green-500" };
  if (active.length <= 2) return { label: "Busy", variant: "secondary", colorClass: "bg-amber-500" };
  return { label: "Overloaded", variant: "destructive", colorClass: "bg-red-500" };
}

function ScoreBar({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-16 text-[10px] font-medium uppercase tracking-wider text-muted-foreground/70">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted/60">
        <div
          className="h-full rounded-full bg-primary/80 transition-all duration-500"
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
      <span className="w-8 text-right text-[10px] font-bold text-foreground/80">{value}%</span>
    </div>
  );
}

export function TeamView({
  members,
  tickets,
  projectSlug,
}: TeamViewProps) {
  const { token } = useAuth();
  const [selectedMember, setSelectedMember] = useState<ProjectMember | null>(null);
  const [selectedMemberIndex, setSelectedMemberIndex] = useState(0);
  const [recommendations, setRecommendations] = useState<EngineerTicketRecommendationsResponse | null>(null);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchRecs() {
      if (!selectedMember || !token) return;

      setRecLoading(true);
      setRecError(null);
      setRecommendations(null);

      const result = await getEngineerTicketRecommendations(
        token,
        projectSlug,
        selectedMember.user_id
      );

      if (result.error) {
        setRecError(result.error);
      } else {
        setRecommendations(result.data);
      }
      setRecLoading(false);
    }

    void fetchRecs();
  }, [selectedMember, token, projectSlug]);

  function handleMemberClick(member: ProjectMember, index: number) {
    setSelectedMember(member);
    setSelectedMemberIndex(index);
  }

  return (
    <>
      {/* ========== TEAM GRID ========== */}
      <div className="p-6">
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {members.map((member, idx) => {
            const assigned = getAssignedTickets(tickets, member.user_id);
            const active = getActiveTickets(tickets, member.user_id);
            const availability = getAvailabilityLabel(tickets, member.user_id);
            const memberColor = AVATAR_COLORS[idx % AVATAR_COLORS.length];

            return (
              <Card
                key={member.id}
                className="group relative flex flex-col cursor-pointer overflow-hidden border-border/50 bg-card transition-all duration-300 hover:-translate-y-[2px] hover:border-primary/30 hover:shadow-lg dark:hover:shadow-none dark:hover:bg-accent/5"
                onClick={() => handleMemberClick(member, idx)}
              >
                {/* Sleek Top Status Bar mapped to Member Color */}
                <div 
                  className="absolute inset-x-0 top-0 h-[3px] transition-opacity opacity-80 group-hover:opacity-100" 
                  style={{ backgroundColor: memberColor }}
                />

                <CardHeader className="p-5 pb-4">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-3.5">
                      <div className="relative shrink-0">
                        <div
                          className="flex size-10 items-center justify-center rounded-full text-sm font-bold text-white shadow-sm ring-1 ring-black/10"
                          style={{ backgroundColor: memberColor }}
                        >
                          {member.first_name[0]}{member.last_name[0]}
                        </div>
                        {/* Status dot remains tied to availability */}
                        <span className={cn("absolute -bottom-0.5 -right-0.5 size-3.5 rounded-full border-2 border-card", availability.colorClass)} />
                      </div>
                      <div className="flex flex-col space-y-0.5">
                        <CardTitle className="text-sm font-bold tracking-tight text-foreground leading-none">
                          {member.first_name} {member.last_name}
                        </CardTitle>
                        <p className="text-xs font-medium text-muted-foreground capitalize">
                          {member.role}
                        </p>
                      </div>
                    </div>
                    <Badge variant={availability.variant} className="shrink-0 text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5">
                      {availability.label}
                    </Badge>
                  </div>
                </CardHeader>

                <CardContent className="flex-1 p-5 pt-0 flex flex-col">
                  {/* Workload Mini-Dashboard */}
                  <div className="mb-4 flex items-center justify-between rounded-lg border border-border/40 bg-muted/30 px-4 py-2.5">
                    <div className="flex flex-col items-start">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70">Active</span>
                      <span className="text-sm font-black text-foreground/90 leading-none mt-1">{active.length}</span>
                    </div>
                    <div className="h-6 w-px bg-border/50" />
                    <div className="flex flex-col items-end">
                      <span className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70">Total</span>
                      <span className="text-sm font-black text-foreground/90 leading-none mt-1">{assigned.length}</span>
                    </div>
                  </div>

                  {/* Tickets List */}
                  <div className="mt-auto">
                    {assigned.length > 0 ? (
                      <div className="space-y-2">
                        {assigned.slice(0, 3).map((ticket) => {
                          const isOverdue = ticket.due_date && new Date(ticket.due_date + "T00:00:00") < new Date(new Date().toISOString().split("T")[0] + "T00:00:00");
                          
                          return (
                            <div key={ticket.id} className="flex items-center gap-2.5 rounded-md border border-border/40 bg-background px-2.5 py-2 shadow-sm transition-colors group-hover:border-border/80">
                              <span className="shrink-0 text-[10px] font-mono font-bold text-muted-foreground/80">
                                {ticket.ticket_key}
                              </span>
                              <span className="truncate text-xs font-medium text-foreground/90">
                                {ticket.title}
                              </span>
                              {isOverdue && (
                                <span className="ml-auto shrink-0 rounded bg-red-500/10 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-red-600 dark:text-red-400">
                                  Late
                                </span>
                              )}
                            </div>
                          );
                        })}
                        {assigned.length > 3 && (
                          <p className="text-[10px] font-bold text-muted-foreground/50 pt-1.5 text-center uppercase tracking-wider">
                            +{assigned.length - 3} more tickets
                          </p>
                        )}
                      </div>
                    ) : (
                      <div className="flex h-[116px] flex-col items-center justify-center rounded-lg border border-dashed border-border/60 bg-muted/10">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/50">Queue Empty</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* ========== MEMBER DETAIL MODAL ========== */}
      <Dialog open={!!selectedMember} onOpenChange={(v) => !v && setSelectedMember(null)}>
        {selectedMember && (
          <DialogContent className="flex max-h-[90vh] w-[96vw] flex-col overflow-hidden p-0 sm:w-[85vw] sm:!max-w-[1100px]">
            
            {/* Header - Fixed */}
            <DialogHeader className="shrink-0 border-b bg-background px-6 py-4 shadow-sm z-10">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-center gap-4">
                  <div className="relative shrink-0">
                    <div
                      className="flex size-12 items-center justify-center rounded-full text-lg font-bold text-white shadow-sm"
                      style={{ backgroundColor: AVATAR_COLORS[selectedMemberIndex % AVATAR_COLORS.length] }}
                    >
                      {selectedMember.first_name[0]}{selectedMember.last_name[0]}
                    </div>
                    <span
                      className={cn(
                        "absolute bottom-0 right-0 size-3.5 rounded-full border-2 border-background",
                        getAvailabilityLabel(tickets, selectedMember.user_id).colorClass
                      )}
                    />
                  </div>
                  <div className="space-y-0.5">
                    <DialogTitle className="text-lg font-semibold tracking-tight">
                      {selectedMember.first_name} {selectedMember.last_name}
                    </DialogTitle>
                    <p className="text-sm font-medium text-muted-foreground">
                      @{selectedMember.username} <span className="mx-1.5 opacity-50">•</span> {selectedMember.email}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2.5">
                  <Badge variant={getAvailabilityLabel(tickets, selectedMember.user_id).variant} className="px-2.5 py-0.5 text-xs">
                    {getAvailabilityLabel(tickets, selectedMember.user_id).label}
                  </Badge>
                  <Badge variant="secondary" className="capitalize px-2.5 py-0.5 text-xs bg-muted text-muted-foreground">
                    {selectedMember.role}
                  </Badge>
                  <div className="ml-2 flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
                    <span className="text-foreground">{getActiveTickets(tickets, selectedMember.user_id).length}</span> active
                  </div>
                </div>
              </div>
            </DialogHeader>

            {/* Body - Flex layout for independent scrolling columns */}
            <div className="flex flex-1 flex-col overflow-hidden lg:flex-row bg-background">
              
              {/* Left Column: Current Tickets */}
              <div className="flex w-full shrink-0 flex-col border-b bg-muted/10 lg:w-[340px] lg:border-b-0 lg:border-r">
                <div className="p-5 border-b border-border/50 bg-muted/5">
                  <h3 className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
                    Assigned Workload ({getAssignedTickets(tickets, selectedMember.user_id).length})
                  </h3>
                </div>
                
                <div className="flex-1 overflow-y-auto p-5">
                  {getAssignedTickets(tickets, selectedMember.user_id).length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-40 text-center text-muted-foreground border border-dashed rounded-lg bg-background/50">
                      <User className="size-8 mb-2 opacity-20" />
                      <p className="text-sm font-medium">No active tickets</p>
                      <p className="text-xs opacity-70">This member's queue is clear.</p>
                    </div>
                  ) : (
                    <div className="space-y-2.5">
                      {getAssignedTickets(tickets, selectedMember.user_id).map((ticket) => {
                        const isOverdue = ticket.due_date && new Date(ticket.due_date + "T00:00:00") < new Date(new Date().toISOString().split("T")[0] + "T00:00:00");

                        return (
                          <div key={ticket.id} className="group flex flex-col gap-2 rounded-lg border bg-background p-3 shadow-sm transition-all hover:shadow-md">
                            <div className="flex items-start justify-between gap-2">
                              <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono font-bold text-muted-foreground">
                                {ticket.ticket_key}
                              </span>
                              <div className="flex flex-wrap shrink-0 items-center gap-1.5 justify-end">
                                {isOverdue && (
                                  <Badge variant="destructive" className="text-[9px] px-1.5 py-0 uppercase tracking-wider font-bold">
                                    Overdue
                                  </Badge>
                                )}
                                <Badge variant="outline" className="text-[9px] px-1.5 py-0 bg-background">
                                  {ticket.priority}
                                </Badge>
                              </div>
                            </div>
                            <span className="text-sm font-medium leading-tight text-foreground/90">
                              {ticket.title}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column: AI Recommendations */}
              <div className="flex flex-1 flex-col overflow-hidden bg-background">
                <div className="flex items-center justify-between border-b border-border/50 p-5 shrink-0 bg-background z-10">
                  <div className="flex items-center gap-2">
                    <Sparkles className="size-4 text-primary" />
                    <h3 className="text-sm font-bold uppercase tracking-wider text-primary">
                      AI Ticket Recommendations
                    </h3>
                  </div>
                  <span className="text-xs text-muted-foreground font-medium">
                    Based on resume & skill match
                  </span>
                </div>

                <div className="flex-1 overflow-y-auto p-5">
                  {recLoading ? (
                    <div className="flex flex-col items-center justify-center h-64 gap-3 text-muted-foreground">
                      <Loader2 className="size-6 animate-spin text-primary" />
                      <p className="text-sm font-medium animate-pulse">Analyzing project backlog & engineer skills...</p>
                    </div>
                  ) : recError ? (
                    <div className="flex flex-col items-center justify-center h-64 rounded-xl border border-dashed bg-destructive/5 p-6 text-center">
                      <p className="text-sm font-semibold text-destructive">{recError}</p>
                    </div>
                  ) : recommendations && recommendations.recommendations.length > 0 ? (
                    <div className="space-y-3 pb-8">
                      {/* Desktop Table Header */}
                      <div className="hidden lg:grid grid-cols-12 gap-4 px-4 text-[10px] font-bold uppercase tracking-wider text-muted-foreground mb-2">
                        <span className="col-span-1 text-center">Match</span>
                        <span className="col-span-5">Ticket Details</span>
                        <span className="col-span-3">Score Breakdown</span>
                        <span className="col-span-2">Current Status</span>
                        <span className="col-span-1"></span>
                      </div>

                      {recommendations.recommendations.map((ticket) => {
                        const score = Math.round((ticket.recommendation_score ?? ticket.similarity_score ?? 0) * 100);
                        const semantic = Math.round((ticket.semantic_similarity ?? 0) * 100);
                        const lexical = Math.round((ticket.lexical_score ?? 0) * 100);

                        return (
                          <div
                            key={ticket.ticket_key}
                            className="group flex flex-col lg:grid lg:grid-cols-12 items-start lg:items-center gap-4 rounded-xl border bg-card p-4 transition-all hover:border-primary/30 hover:shadow-md"
                          >
                            {/* Score */}
                            <div className="col-span-1 flex items-baseline justify-center shrink-0 bg-primary/10 rounded-lg p-2 lg:p-0 lg:bg-transparent lg:rounded-none min-w-[60px]">
                              <span className="text-2xl font-black tracking-tighter text-primary">
                                {score}
                              </span>
                              <span className="text-xs font-bold text-primary/70 ml-0.5">%</span>
                            </div>

                            {/* Ticket Info */}
                            <div className="col-span-5 min-w-0 w-full space-y-2">
                              <div className="flex items-start gap-2.5">
                                <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] font-bold text-muted-foreground mt-0.5">
                                  {ticket.ticket_key}
                                </span>
                                <span className="text-sm font-semibold leading-snug line-clamp-2 group-hover:text-primary transition-colors">
                                  {ticket.title}
                                </span>
                              </div>
                              <div className="flex flex-wrap items-center gap-1.5">
                                <Badge variant="outline" className="text-[9px] bg-background">
                                  {ticket.priority}
                                </Badge>
                                <Badge variant="secondary" className="text-[9px] bg-muted/50">
                                  {ticket.type}
                                </Badge>
                                {ticket.labels?.slice(0, 3).map((label) => (
                                  <span key={label} className="text-[10px] text-muted-foreground flex items-center gap-1 border rounded px-1.5 py-0.5 bg-muted/10">
                                    <span className="size-1.5 rounded-full bg-primary/40"></span>
                                    {label}
                                  </span>
                                ))}
                              </div>
                            </div>

                            {/* Score Breakdown */}
                            <div className="col-span-3 w-full space-y-2.5 border-t lg:border-t-0 pt-3 lg:pt-0 border-border/50">
                              <ScoreBar value={semantic} label="Semantic" />
                              <ScoreBar value={lexical} label="Lexical" />
                            </div>

                            {/* Status */}
                            <div className="col-span-2 w-full flex flex-row lg:flex-col items-center lg:items-start justify-between lg:justify-center border-t lg:border-t-0 pt-3 lg:pt-0 border-border/50">
                              <div className="flex items-center gap-2">
                                <div className="size-2 rounded-full bg-blue-500/50"></div>
                                <span className="text-xs font-medium">{ticket.column_name}</span>
                              </div>
                              {ticket.assignee_name && (
                                <p className="text-[10px] font-medium text-muted-foreground/80 mt-1 lg:ml-4">
                                  Assigned: {ticket.assignee_name}
                                </p>
                              )}
                            </div>

                            {/* Action */}
                            <div className="col-span-1 w-full lg:w-auto flex justify-end mt-2 lg:mt-0">
                              <Button
                                size="sm"
                                variant="secondary"
                                className="w-full lg:w-9 lg:h-9 lg:p-0 rounded-full lg:opacity-0 lg:group-hover:opacity-100 transition-all hover:bg-primary hover:text-primary-foreground"
                              >
                                <span className="lg:hidden mr-2">View Ticket</span>
                                <ArrowRight className="size-4" />
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-64 rounded-xl border border-dashed bg-muted/10 p-6 text-center">
                      <div className="flex size-12 items-center justify-center rounded-full bg-muted mb-4">
                        <Sparkles className="size-6 text-muted-foreground/40" />
                      </div>
                      <p className="text-base font-semibold text-foreground/80">
                        No recommendations available
                      </p>
                      <p className="mt-1 text-sm text-muted-foreground max-w-sm">
                        Ensure the project backlog has active tickets and the engineer profile contains up-to-date resume data.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </DialogContent>
        )}
      </Dialog>
    </>
  );
}