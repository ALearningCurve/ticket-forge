"use client";

import { useEffect, useState } from "react";
import { Loader2, Sparkles, ArrowRight, User, Search } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
import { TicketDetailModal } from "@/components/projects/board/ticket-detail-modal";
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
  boardColumns: { id: string; name: string }[];
  sizePointsMap: { S: number; M: number; L: number; XL: number };
  weeklyPointsPerMember: number;
}

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

const DONE_COLUMNS = new Set([
  "done",
  "complete",
  "completed",
  "closed",
  "resolved",
]);

function getOpenTickets(
  tickets: TicketResponse[],
  userId: string,
  boardColumns: { id: string; name: string }[]
): TicketResponse[] {
  return tickets.filter((t) => {
    if (t.assignee?.id !== userId) return false;
    const col = boardColumns.find((c) => c.id === t.column_id);
    return !col || !DONE_COLUMNS.has(col.name.toLowerCase());
  });
}

function getDoneTickets(
  tickets: TicketResponse[],
  userId: string,
  boardColumns: { id: string; name: string }[]
): TicketResponse[] {
  return tickets.filter((t) => {
    if (t.assignee?.id !== userId) return false;
    const col = boardColumns.find((c) => c.id === t.column_id);
    return col != null && DONE_COLUMNS.has(col.name.toLowerCase());
  });
}

function getAvailabilityLabel(
  tickets: TicketResponse[],
  userId: string,
  boardColumns: { id: string; name: string }[]
): {
  label: string;
  variant: "default" | "destructive" | "secondary";
  colorClass: string;
} {
  const open = getOpenTickets(tickets, userId, boardColumns);
  if (open.length === 0)
    return {
      label: "Available",
      variant: "default",
      colorClass: "bg-green-500",
    };
  if (open.length <= 2)
    return { label: "Busy", variant: "secondary", colorClass: "bg-amber-500" };
  return {
    label: "Overloaded",
    variant: "destructive",
    colorClass: "bg-red-500",
  };
}

function ScoreBar({ value, label }: { value: number; label: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-16 text-[10px] font-medium uppercase tracking-wider text-muted-foreground/70">
        {label}
      </span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted/60">
        <div
          className="h-full rounded-full bg-primary/80 transition-all duration-500"
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
      <span className="w-8 text-right text-[10px] font-bold text-foreground/80">
        {value}%
      </span>
    </div>
  );
}

export function TeamView({
  members,
  tickets,
  projectSlug,
  boardColumns,
  sizePointsMap: _sizePointsMap,
  weeklyPointsPerMember: _weeklyPointsPerMember,
}: TeamViewProps) {
  const { token } = useAuth();
  const [selectedMember, setSelectedMember] =
    useState<ProjectMember | null>(null);
  const [selectedMemberIndex, setSelectedMemberIndex] = useState(0);
  const [recommendations, setRecommendations] =
    useState<EngineerTicketRecommendationsResponse | null>(null);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);
  const [viewTicketKey, setViewTicketKey] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const filteredMembers = searchQuery
    ? members.filter((m) => {
        const q = searchQuery.toLowerCase();
        return (
          m.first_name.toLowerCase().includes(q) ||
          m.last_name.toLowerCase().includes(q) ||
          m.username.toLowerCase().includes(q) ||
          m.email.toLowerCase().includes(q)
        );
      })
    : members;

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
        {/* Search & Count */}
        <div className="mb-5 flex items-center justify-between gap-4">
          <div className="relative w-64">
            <Search className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search team members..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 pl-8 text-[13px]"
            />
          </div>
          <span className="text-xs text-muted-foreground">
            {filteredMembers.length} of {members.length} members
          </span>
        </div>

        {filteredMembers.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed py-16 text-center">
            <User className="mb-3 size-8 text-muted-foreground/30" />
            <p className="text-sm font-medium text-muted-foreground">
              No members match &ldquo;{searchQuery}&rdquo;
            </p>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
            {filteredMembers.map((member, idx) => {
              const openTickets = getOpenTickets(
                tickets,
                member.user_id,
                boardColumns
              );
              const doneTickets = getDoneTickets(
                tickets,
                member.user_id,
                boardColumns
              );
              const availability = getAvailabilityLabel(
                tickets,
                member.user_id,
                boardColumns
              );
              const memberColor = AVATAR_COLORS[idx % AVATAR_COLORS.length];

              return (
                <Card
                  key={member.id}
                  className="group relative flex cursor-pointer flex-col overflow-hidden border-border/60 bg-card transition-all duration-200 hover:-translate-y-[1px] hover:border-primary/30 hover:shadow-md"
                  onClick={() => handleMemberClick(member, idx)}
                >
                  {/* Sleek Top Line */}
                  <div
                    className="absolute inset-x-0 top-0 h-[2px] opacity-80 transition-opacity group-hover:opacity-100"
                    style={{ backgroundColor: memberColor }}
                  />

                  <div className="flex flex-col gap-3 p-4">
                    {/* Header Row */}
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <div className="relative shrink-0">
                          <div
                            className="flex size-8 items-center justify-center rounded-full text-xs font-bold text-white shadow-sm ring-1 ring-black/10"
                            style={{ backgroundColor: memberColor }}
                          >
                            {member.first_name[0]}
                            {member.last_name[0]}
                          </div>
                          <span
                            className={cn(
                              "absolute -bottom-0.5 -right-0.5 size-2.5 rounded-full border-2 border-card",
                              availability.colorClass
                            )}
                          />
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[13px] font-semibold leading-tight tracking-tight text-foreground">
                            {member.first_name} {member.last_name}
                          </span>
                          <span className="text-[11px] font-medium capitalize text-muted-foreground">
                            {member.role}
                          </span>
                        </div>
                      </div>
                      <Badge
                        variant={availability.variant}
                        className="h-4 shrink-0 px-1.5 py-0 text-[9px] font-bold uppercase tracking-wider"
                      >
                        {availability.label}
                      </Badge>
                    </div>

                    {/* Compact Stats Row */}
                    <div className="mt-0.5 flex items-center gap-4 text-[11px] font-medium text-muted-foreground">
                      <div className="flex items-center gap-1.5">
                        <span className="size-1.5 rounded-full bg-primary/60" />
                        <span className="font-bold text-foreground/90">
                          {openTickets.length}
                        </span>{" "}
                        Open
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className="size-1.5 rounded-full bg-green-500/60" />
                        <span className="font-bold text-foreground/90">
                          {doneTickets.length}
                        </span>{" "}
                        Done
                      </div>
                    </div>

                    {/* Subtle Separator */}
                    <div className="my-0.5 h-px w-full bg-border/40" />

                    {/* Compact Tickets List or AI Teaser */}
                    <div className="flex-1">
                      {openTickets.length > 0 ? (
                        <div className="space-y-1">
                          {openTickets.slice(0, 3).map((ticket) => {
                            const isOverdue =
                              ticket.due_date &&
                              new Date(ticket.due_date + "T00:00:00") 
                                new Date(
                                  new Date().toISOString().split("T")[0] +
                                    "T00:00:00"
                                );
                            return (
                              <div
                                key={ticket.id}
                                className="flex items-center gap-2 rounded px-1.5 py-1 transition-colors hover:bg-muted/50"
                              >
                                <span className="shrink-0 text-[9px] font-mono font-medium text-muted-foreground/60">
                                  {ticket.ticket_key}
                                </span>
                                <span className="truncate text-[11px] font-medium text-foreground/80">
                                  {ticket.title}
                                </span>
                                {isOverdue && (
                                  <span
                                    className="ml-auto size-1.5 shrink-0 rounded-full bg-red-500 shadow-sm"
                                    title="Overdue ticket"
                                  />
                                )}
                              </div>
                            );
                          })}
                          {openTickets.length > 3 && (
                            <div className="pl-1.5 pt-1.5 text-[9px] font-semibold uppercase tracking-wider text-muted-foreground/50">
                              +{openTickets.length - 3} more
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 rounded-md border border-dashed border-primary/20 bg-primary/5 px-2.5 py-2.5">
                          <Sparkles className="size-3.5 shrink-0 text-primary/60" />
                          <span className="text-[10px] font-medium text-primary/70">
                            Click to see AI recommendations
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </div>

      {/* ========== MEMBER DETAIL MODAL ========== */}
      <Dialog
        open={!!selectedMember}
        onOpenChange={(v) => !v && setSelectedMember(null)}
      >
        {selectedMember && (
          <DialogContent className="flex max-h-[90vh] w-[96vw] flex-col overflow-hidden p-0 sm:w-[85vw] sm:!max-w-[1100px]">
            {/* Header - Fixed */}
            <DialogHeader className="z-10 shrink-0 border-b bg-background px-6 py-4 shadow-sm">
              <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                <div className="flex items-center gap-4">
                  <div className="relative shrink-0">
                    <div
                      className="flex size-12 items-center justify-center rounded-full text-lg font-bold text-white shadow-sm"
                      style={{
                        backgroundColor:
                          AVATAR_COLORS[
                            selectedMemberIndex % AVATAR_COLORS.length
                          ],
                      }}
                    >
                      {selectedMember.first_name[0]}
                      {selectedMember.last_name[0]}
                    </div>
                    <span
                      className={cn(
                        "absolute bottom-0 right-0 size-3.5 rounded-full border-2 border-background",
                        getAvailabilityLabel(
                          tickets,
                          selectedMember.user_id,
                          boardColumns
                        ).colorClass
                      )}
                    />
                  </div>
                  <div className="space-y-0.5">
                    <DialogTitle className="text-lg font-semibold tracking-tight">
                      {selectedMember.first_name} {selectedMember.last_name}
                    </DialogTitle>
                    <p className="text-sm font-medium text-muted-foreground">
                      @{selectedMember.username}{" "}
                      <span className="mx-1.5 opacity-50">•</span>{" "}
                      {selectedMember.email}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2.5">
                  <Badge
                    variant={
                      getAvailabilityLabel(
                        tickets,
                        selectedMember.user_id,
                        boardColumns
                      ).variant
                    }
                    className="px-2.5 py-0.5 text-xs"
                  >
                    {
                      getAvailabilityLabel(
                        tickets,
                        selectedMember.user_id,
                        boardColumns
                      ).label
                    }
                  </Badge>
                  <Badge
                    variant="secondary"
                    className="bg-muted px-2.5 py-0.5 text-xs capitalize text-muted-foreground"
                  >
                    {selectedMember.role}
                  </Badge>
                  <div className="ml-2 flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
                    <span className="text-foreground">
                      {
                        getOpenTickets(
                          tickets,
                          selectedMember.user_id,
                          boardColumns
                        ).length
                      }
                    </span>{" "}
                    active
                  </div>
                </div>
              </div>
            </DialogHeader>

            {/* Body - Flex layout for independent scrolling columns */}
            <div className="flex flex-1 flex-col overflow-hidden bg-background lg:flex-row">
              {/* Left Column: Open Tickets */}
              <div className="flex w-full shrink-0 flex-col border-b bg-muted/10 lg:w-[340px] lg:border-b-0 lg:border-r">
                <div className="border-b border-border/50 bg-muted/5 p-5">
                  <h3 className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground">
                    Active Workload (
                    {
                      getOpenTickets(
                        tickets,
                        selectedMember.user_id,
                        boardColumns
                      ).length
                    }
                    )
                  </h3>
                </div>

                <div className="flex-1 overflow-y-auto p-5">
                  {getOpenTickets(
                    tickets,
                    selectedMember.user_id,
                    boardColumns
                  ).length === 0 ? (
                    <div className="flex h-40 flex-col items-center justify-center rounded-lg border border-dashed bg-background/50 text-center text-muted-foreground">
                      <User className="mb-2 size-8 opacity-20" />
                      <p className="text-sm font-medium">No active tickets</p>
                      <p className="text-xs opacity-70">
                        This member&apos;s queue is clear.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2.5">
                      {getOpenTickets(
                        tickets,
                        selectedMember.user_id,
                        boardColumns
                      ).map((ticket) => {
                        const isOverdue =
                          ticket.due_date &&
                          new Date(ticket.due_date + "T00:00:00") 
                            new Date(
                              new Date().toISOString().split("T")[0] +
                                "T00:00:00"
                            );

                        return (
                          <div
                            key={ticket.id}
                            className="group flex flex-col gap-2 rounded-lg border bg-background p-3 shadow-sm transition-all hover:shadow-md"
                          >
                            <div className="flex items-start justify-between gap-2">
                              <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] font-bold text-muted-foreground">
                                {ticket.ticket_key}
                              </span>
                              <div className="flex shrink-0 flex-wrap items-center justify-end gap-1.5">
                                {isOverdue && (
                                  <Badge
                                    variant="destructive"
                                    className="px-1.5 py-0 text-[9px] font-bold uppercase tracking-wider"
                                  >
                                    Overdue
                                  </Badge>
                                )}
                                <Badge
                                  variant="outline"
                                  className="bg-background px-1.5 py-0 text-[9px]"
                                >
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
                <div className="z-10 flex shrink-0 items-center justify-between border-b border-border/50 bg-background p-5">
                  <div className="flex items-center gap-2">
                    <Sparkles className="size-4 text-primary" />
                    <h3 className="text-sm font-bold uppercase tracking-wider text-primary">
                      AI Ticket Recommendations
                    </h3>
                  </div>
                  <span className="text-xs font-medium text-muted-foreground">
                    Based on resume & skill match
                  </span>
                </div>

                <div className="flex-1 overflow-y-auto p-5">
                  {recLoading ? (
                    <div className="flex h-64 flex-col items-center justify-center gap-3 text-muted-foreground">
                      <Loader2 className="size-6 animate-spin text-primary" />
                      <p className="animate-pulse text-sm font-medium">
                        Analyzing project backlog & engineer skills...
                      </p>
                    </div>
                  ) : recError ? (
                    <div className="flex h-64 flex-col items-center justify-center rounded-xl border border-dashed bg-destructive/5 p-6 text-center">
                      <p className="text-sm font-semibold text-destructive">
                        {recError}
                      </p>
                    </div>
                  ) : recommendations &&
                    recommendations.recommendations.length > 0 ? (
                    <div className="space-y-3 pb-8">
                      {/* Desktop Table Header */}
                      <div className="mb-2 hidden grid-cols-12 gap-4 px-4 text-[10px] font-bold uppercase tracking-wider text-muted-foreground lg:grid">
                        <span className="col-span-1 text-center">Match</span>
                        <span className="col-span-5">Ticket Details</span>
                        <span className="col-span-3">Score Breakdown</span>
                        <span className="col-span-2">Current Status</span>
                        <span className="col-span-1"></span>
                      </div>

                      {recommendations.recommendations.map((ticket) => {
                        const score = Math.round(
                          (ticket.recommendation_score ??
                            ticket.similarity_score ??
                            0) * 100
                        );
                        const semantic = Math.round(
                          (ticket.semantic_similarity ?? 0) * 100
                        );
                        const lexical = Math.round(
                          (ticket.lexical_score ?? 0) * 100
                        );

                        return (
                          <div
                            key={ticket.ticket_key}
                            className="group flex flex-col items-start gap-4 rounded-xl border bg-card p-4 transition-all hover:border-primary/30 hover:shadow-md lg:grid lg:grid-cols-12 lg:items-center"
                          >
                            {/* Score */}
                            <div className="col-span-1 flex min-w-[60px] shrink-0 items-baseline justify-center rounded-lg bg-primary/10 p-2 lg:rounded-none lg:bg-transparent lg:p-0">
                              <span className="text-2xl font-black tracking-tighter text-primary">
                                {score}
                              </span>
                              <span className="ml-0.5 text-xs font-bold text-primary/70">
                                %
                              </span>
                            </div>

                            {/* Ticket Info */}
                            <div className="col-span-5 min-w-0 w-full space-y-2">
                              <div className="flex items-start gap-2.5">
                                <span className="mt-0.5 shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] font-bold text-muted-foreground">
                                  {ticket.ticket_key}
                                </span>
                                <span className="line-clamp-2 text-sm font-semibold leading-snug transition-colors group-hover:text-primary">
                                  {ticket.title}
                                </span>
                              </div>
                              <div className="flex flex-wrap items-center gap-1.5">
                                <Badge
                                  variant="outline"
                                  className="bg-background text-[9px]"
                                >
                                  {ticket.priority}
                                </Badge>
                                <Badge
                                  variant="secondary"
                                  className="bg-muted/50 text-[9px]"
                                >
                                  {ticket.type}
                                </Badge>
                                {ticket.size_bucket && (
                                  <Badge
                                    variant="secondary"
                                    className="bg-muted/50 text-[9px]"
                                  >
                                    {ticket.size_bucket}
                                  </Badge>
                                )}
                                {ticket.labels?.slice(0, 3).map((label) => (
                                  <span
                                    key={label}
                                    className="flex items-center gap-1 rounded border bg-muted/10 px-1.5 py-0.5 text-[10px] text-muted-foreground"
                                  >
                                    <span className="size-1.5 rounded-full bg-primary/40"></span>
                                    {label}
                                  </span>
                                ))}
                              </div>
                            </div>

                            {/* Score Breakdown */}
                            <div className="col-span-3 w-full space-y-2.5 border-t border-border/50 pt-3 lg:border-t-0 lg:pt-0">
                              <ScoreBar value={semantic} label="Semantic" />
                              <ScoreBar value={lexical} label="Lexical" />
                            </div>

                            {/* Status */}
                            <div className="col-span-2 flex w-full flex-row items-center justify-between border-t border-border/50 pt-3 lg:flex-col lg:items-start lg:justify-center lg:border-t-0 lg:pt-0">
                              <div className="flex items-center gap-2">
                                <div className="size-2 rounded-full bg-blue-500/50"></div>
                                <span className="text-xs font-medium">
                                  {ticket.column_name}
                                </span>
                              </div>
                              {ticket.assignee_name && (
                                <p className="mt-1 text-[10px] font-medium text-muted-foreground/80 lg:ml-4">
                                  Assigned: {ticket.assignee_name}
                                </p>
                              )}
                            </div>

                            {/* Action */}
                            <div className="col-span-1 mt-2 flex w-full justify-end lg:mt-0 lg:w-auto">
                              <Button
                                size="sm"
                                variant="secondary"
                                className="w-full rounded-full transition-all hover:bg-primary hover:text-primary-foreground lg:h-9 lg:w-9 lg:p-0 lg:opacity-0 lg:group-hover:opacity-100"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setViewTicketKey(ticket.ticket_key);
                                }}
                              >
                                <span className="mr-2 lg:hidden">
                                  View Ticket
                                </span>
                                <ArrowRight className="size-4" />
                              </Button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="flex h-64 flex-col items-center justify-center rounded-xl border border-dashed bg-muted/10 p-6 text-center">
                      <div className="mb-4 flex size-12 items-center justify-center rounded-full bg-muted">
                        <Sparkles className="size-6 text-muted-foreground/40" />
                      </div>
                      <p className="text-base font-semibold text-foreground/80">
                        No recommendations available
                      </p>
                      <p className="mt-1 max-w-sm text-sm text-muted-foreground">
                        Ensure the project backlog has active tickets and the
                        engineer profile contains up-to-date resume data.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </DialogContent>
        )}
      </Dialog>

      {/* Ticket Detail Modal — opened from recommendation arrow */}
      <TicketDetailModal
        key={viewTicketKey || "team-ticket-modal"}
        ticket={
          viewTicketKey
            ? (tickets.find((t) => t.ticket_key === viewTicketKey) ?? null)
            : null
        }
        projectSlug={projectSlug}
        members={members}
        open={!!viewTicketKey}
        onClose={() => setViewTicketKey(null)}
        onUpdated={() => setViewTicketKey(null)}
        onDeleted={() => setViewTicketKey(null)}
      />
    </>
  );
}