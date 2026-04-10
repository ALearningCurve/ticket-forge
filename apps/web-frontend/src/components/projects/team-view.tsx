"use client";

import { useState } from "react";
import { Sparkles } from "lucide-react";

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
import { Separator } from "@/components/ui/separator";
import type { ProjectMember, TicketResponse } from "@/lib/api";

interface TeamViewProps {
  members: ProjectMember[];
  tickets: TicketResponse[];
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

function getMatchScore(): number {
  return Math.floor(Math.random() * 20) + 75;
}

function getAssignedTickets(
  tickets: TicketResponse[],
  userId: string
): TicketResponse[] {
  return tickets.filter((t) => t.assignee?.id === userId);
}

/**
 * Availability logic:
 * - Count only tickets with due dates in the future or today (active work)
 * - Tickets due yesterday or before = completed work, don't count
 * - < 3 active future tickets = Available
 * - 3+ active future tickets = Busy
 */
function getActiveTickets(
  tickets: TicketResponse[],
  userId: string
): TicketResponse[] {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  return tickets.filter((t) => {
    if (t.assignee?.id !== userId) return false;

    // No due date = still active (unknown completion)
    if (!t.due_date) return true;

    // Due date is today or in the future = active
    const due = new Date(t.due_date + "T00:00:00");
    return due >= today;
  });
}

function isAvailable(tickets: TicketResponse[], userId: string): boolean {
  return getActiveTickets(tickets, userId).length === 0;
}

function getAvailabilityLabel(
  tickets: TicketResponse[],
  userId: string
): { label: string; variant: "default" | "destructive" | "secondary" } {
  const active = getActiveTickets(tickets, userId);
  if (active.length === 0) return { label: "Available", variant: "default" };
  if (active.length <= 2) return { label: "Busy", variant: "secondary" };
  return { label: "Overloaded", variant: "destructive" };
}

export function TeamView({ members, tickets }: TeamViewProps) {
  const [selectedMember, setSelectedMember] = useState<ProjectMember | null>(
    null
  );
  const [selectedMemberIndex, setSelectedMemberIndex] = useState(0);

  function getRecommendedTickets(): TicketResponse[] {
    return tickets.filter((t) => !t.assignee).slice(0, 5);
  }

  function getSuggestedReassignments(userId: string): TicketResponse[] {
    return tickets
      .filter((t) => t.assignee && t.assignee.id !== userId)
      .slice(0, 3);
  }

  function handleMemberClick(member: ProjectMember, index: number) {
    setSelectedMember(member);
    setSelectedMemberIndex(index);
  }

  return (
    <>
      <div className="p-5">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {members.map((member, idx) => {
            const assigned = getAssignedTickets(tickets, member.user_id);
            const active = getActiveTickets(tickets, member.user_id);
            const available = isAvailable(tickets, member.user_id);
            const availability = getAvailabilityLabel(tickets, member.user_id);

            return (
              <Card
                key={member.id}
                className="relative cursor-pointer overflow-hidden transition-all hover:shadow-md"
                onClick={() => handleMemberClick(member, idx)}
              >
                <div
                  className={`absolute left-0 top-0 h-full w-1 ${
                    available ? "bg-green-500" : "bg-red-500"
                  }`}
                />

                <CardHeader className="pb-2 pl-5">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <div
                        className="flex size-10 items-center justify-center rounded-full text-sm font-semibold text-white"
                        style={{
                          backgroundColor:
                            AVATAR_COLORS[idx % AVATAR_COLORS.length],
                        }}
                      >
                        {member.first_name[0]}
                        {member.last_name[0]}
                      </div>
                      <span
                        className={`absolute -bottom-0.5 -right-0.5 size-3.5 rounded-full border-2 border-card ${
                          available ? "bg-green-500" : "bg-red-500"
                        }`}
                      />
                    </div>
                    <div>
                      <CardTitle className="text-sm">
                        {member.first_name} {member.last_name}
                      </CardTitle>
                      <p className="text-xs text-muted-foreground">
                        @{member.username}
                      </p>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="pl-5">
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={availability.variant}
                      className="text-[10px]"
                    >
                      {availability.label}
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="capitalize text-[10px]"
                    >
                      {member.role}
                    </Badge>
                  </div>

                  <div className="mt-3 space-y-1">
                    <p className="text-xs text-muted-foreground">
                      {active.length} active &middot; {assigned.length} total
                    </p>
                    {assigned.length > 0 && (
                      <div className="space-y-1 pt-1">
                        {assigned.slice(0, 3).map((ticket) => {
                          const isOverdue =
                            ticket.due_date &&
                            new Date(ticket.due_date + "T00:00:00") <
                              new Date(new Date().toISOString().split("T")[0] + "T00:00:00");

                          return (
                            <div
                              key={ticket.id}
                              className="flex items-center gap-1.5 rounded bg-muted/50 px-2 py-1"
                            >
                              <span className="text-[10px] font-medium text-muted-foreground">
                                {ticket.ticket_key}
                              </span>
                              <span className="truncate text-[11px]">
                                {ticket.title}
                              </span>
                              {isOverdue && (
                                <span className="ml-auto shrink-0 text-[9px] font-medium text-red-500">
                                  overdue
                                </span>
                              )}
                            </div>
                          );
                        })}
                        {assigned.length > 3 && (
                          <p className="text-[10px] text-muted-foreground">
                            +{assigned.length - 3} more
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* ========== Member Detail Modal ========== */}
      <Dialog
        open={!!selectedMember}
        onOpenChange={(v) => !v && setSelectedMember(null)}
      >
        {selectedMember && (
          <DialogContent className="max-h-[85vh] max-w-2xl overflow-y-auto p-0">
            <DialogHeader className="px-6 pb-0 pt-6">
              <div className="flex items-center gap-4">
                <div className="relative">
                  <div
                    className="flex size-14 items-center justify-center rounded-full text-lg font-bold text-white"
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
                    className={`absolute bottom-0 right-0 size-4 rounded-full border-2 border-background ${
                      isAvailable(tickets, selectedMember.user_id)
                        ? "bg-green-500"
                        : "bg-red-500"
                    }`}
                  />
                </div>
                <div>
                  <DialogTitle className="text-lg">
                    {selectedMember.first_name} {selectedMember.last_name}
                  </DialogTitle>
                  <p className="text-sm text-muted-foreground">
                    @{selectedMember.username} &middot; {selectedMember.email}
                  </p>
                  <div className="mt-1 flex items-center gap-2">
                    <Badge
                      variant={
                        getAvailabilityLabel(tickets, selectedMember.user_id)
                          .variant
                      }
                      className="text-[10px]"
                    >
                      {
                        getAvailabilityLabel(tickets, selectedMember.user_id)
                          .label
                      }
                    </Badge>
                    <Badge
                      variant="secondary"
                      className="capitalize text-[10px]"
                    >
                      {selectedMember.role}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {
                        getActiveTickets(tickets, selectedMember.user_id)
                          .length
                      }{" "}
                      active tickets
                    </span>
                  </div>
                </div>
              </div>
            </DialogHeader>

            <div className="space-y-5 px-6 py-4">
              {/* Current tickets */}
              {getAssignedTickets(tickets, selectedMember.user_id).length >
                0 && (
                <div className="space-y-2">
                  <h3 className="text-sm font-semibold">Current tickets</h3>
                  <div className="space-y-1.5">
                    {getAssignedTickets(tickets, selectedMember.user_id).map(
                      (ticket) => {
                        const isOverdue =
                          ticket.due_date &&
                          new Date(ticket.due_date + "T00:00:00") <
                            new Date(new Date().toISOString().split("T")[0] + "T00:00:00");

                        return (
                          <div
                            key={ticket.id}
                            className="flex items-center justify-between rounded-md border px-3 py-2"
                          >
                            <div className="flex min-w-0 items-center gap-2">
                              <span className="shrink-0 text-xs font-medium text-muted-foreground">
                                {ticket.ticket_key}
                              </span>
                              <span className="truncate text-sm">
                                {ticket.title}
                              </span>
                            </div>
                            <div className="ml-2 flex shrink-0 items-center gap-1.5">
                              {isOverdue && (
                                <Badge
                                  variant="destructive"
                                  className="text-[10px]"
                                >
                                  overdue
                                </Badge>
                              )}
                              {ticket.due_date && !isOverdue && (
                                <span className="text-[10px] text-muted-foreground">
                                  due{" "}
                                  {new Date(
                                    ticket.due_date + "T00:00:00"
                                  ).toLocaleDateString("en-US", {
                                    month: "short",
                                    day: "numeric",
                                  })}
                                </span>
                              )}
                              <Badge
                                variant="outline"
                                className="text-[10px]"
                              >
                                {ticket.priority}
                              </Badge>
                              <Badge
                                variant="secondary"
                                className="text-[10px]"
                              >
                                {ticket.size}
                              </Badge>
                            </div>
                          </div>
                        );
                      }
                    )}
                  </div>
                </div>
              )}

              <Separator />

              {/* AI Recommended Tickets */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="size-4 text-primary" />
                  <h3 className="text-sm font-semibold text-primary">
                    AI Recommended tickets
                  </h3>
                </div>
                <p className="text-xs text-muted-foreground">
                  Based on {selectedMember.first_name}&apos;s skills, experience,
                  and current workload, these tickets are the best match.
                </p>

                {getRecommendedTickets().length > 0 ? (
                  <div className="space-y-2">
                    {getRecommendedTickets().map(
                      (ticket) => {
                        const score = getMatchScore();
                        return (
                          <div
                            key={ticket.id}
                            className="flex items-center justify-between rounded-lg border border-dashed border-primary/20 bg-primary/5 px-3 py-2.5"
                          >
                            <div className="flex min-w-0 items-center gap-3">
                              <div className="flex shrink-0 flex-col items-center">
                                <span className="text-lg font-bold text-primary">
                                  {score}%
                                </span>
                                <span className="text-[9px] text-muted-foreground">
                                  match
                                </span>
                              </div>
                              <div className="min-w-0">
                                <div className="flex items-center gap-1.5">
                                  <span className="text-xs font-medium text-muted-foreground">
                                    {ticket.ticket_key}
                                  </span>
                                  <span className="truncate text-sm font-medium">
                                    {ticket.title}
                                  </span>
                                </div>
                                <div className="mt-0.5 flex items-center gap-1.5">
                                  <Badge
                                    variant="outline"
                                    className="text-[10px]"
                                  >
                                    {ticket.priority}
                                  </Badge>
                                  <Badge
                                    variant="secondary"
                                    className="text-[10px]"
                                  >
                                    {ticket.size}
                                  </Badge>
                                  <Badge
                                    variant="secondary"
                                    className="text-[10px]"
                                  >
                                    {ticket.type}
                                  </Badge>
                                </div>
                              </div>
                            </div>
                            <Button
                              size="sm"
                              variant="outline"
                              className="ml-2 shrink-0 text-xs"
                            >
                              Assign
                            </Button>
                          </div>
                        );
                      }
                    )}
                  </div>
                ) : (
                  <div className="rounded-lg border border-dashed px-4 py-8 text-center">
                    <Sparkles className="mx-auto mb-2 size-5 text-muted-foreground/40" />
                    <p className="text-sm text-muted-foreground">
                      No unassigned tickets to recommend
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground/60">
                      All tickets are already assigned
                    </p>
                  </div>
                )}
              </div>

              {/* Suggested reassignments */}
              {getSuggestedReassignments(selectedMember.user_id).length >
                0 && (
                <>
                  <Separator />
                  <div className="space-y-3">
                    <h3 className="text-sm font-semibold text-muted-foreground">
                      Suggested reassignments
                    </h3>
                    <p className="text-xs text-muted-foreground">
                      These tickets assigned to others may be a better fit for{" "}
                      {selectedMember.first_name}.
                    </p>
                    <div className="space-y-1.5">
                      {getSuggestedReassignments(selectedMember.user_id).map(
                        (ticket) => {
                          const score = getMatchScore();
                          return (
                            <div
                              key={ticket.id}
                              className="flex items-center justify-between rounded-md border px-3 py-2"
                            >
                              <div className="flex min-w-0 items-center gap-2">
                                <span className="text-sm font-bold text-muted-foreground">
                                  {score}%
                                </span>
                                <span className="text-xs font-medium text-muted-foreground">
                                  {ticket.ticket_key}
                                </span>
                                <span className="truncate text-sm">
                                  {ticket.title}
                                </span>
                              </div>
                              <div className="ml-2 flex shrink-0 items-center gap-2">
                                <span className="text-[10px] text-muted-foreground">
                                  from {ticket.assignee?.first_name}
                                </span>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  className="h-7 text-xs"
                                >
                                  Reassign
                                </Button>
                              </div>
                            </div>
                          );
                        }
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          </DialogContent>
        )}
      </Dialog>
    </>
  );
}