"use client";

import { useState, useCallback, useEffect } from "react";
import { DragDropContext, type DropResult } from "@hello-pangea/dnd";
import { Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";

import { BoardColumn, type ColumnData } from "./board-column";
import type { TicketData } from "./board-card";
import { TicketDetailModal } from "./ticket-detail-modal";
import { cn } from "@/lib/utils";
import {
  type BoardColumn as ApiBoardColumn,
  type ProjectMember,
  type TicketResponse,
  classifyMissingTicketSizes,
  getBoardTickets,
  createTicket as apiCreateTicket,
  moveTicket as apiMoveTicket,
} from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

interface BoardViewProps {
  projectSlug: string;
  boardColumns: ApiBoardColumn[];
  members: ProjectMember[];
  ticketFilter?: (ticket: TicketResponse) => boolean;
  sizePointsMap?: { S: number; M: number; L: number; XL: number };
  weeklyPointsPerMember?: number;
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

function getAvatarColor(index: number) {
  return AVATAR_COLORS[index % AVATAR_COLORS.length];
}

function apiTicketToCard(
  t: TicketResponse,
  memberIndex: Map<string, number>,
): TicketData {
  return {
    id: t.id,
    key: t.ticket_key,
    title: t.title,
    type: t.type as TicketData["type"],
    priority: t.priority as TicketData["priority"],
    size: (t.size_bucket || t.size || null) as TicketData["size"],
    labels: t.labels || [],
    dueDate: t.due_date
      ? new Date(t.due_date + "T00:00:00").toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "numeric",
        })
      : undefined,
    assignee: t.assignee
      ? {
          initials: `${t.assignee.first_name[0]}${t.assignee.last_name[0]}`,
          name: `${t.assignee.first_name} ${t.assignee.last_name}`,
          color: getAvatarColor(memberIndex.get(t.assignee.id) ?? 0),
        }
      : undefined,
  };
}

function buildColumns(
  boardColumns: ApiBoardColumn[],
  tickets: TicketResponse[],
  memberIndex: Map<string, number>,
): ColumnData[] {
  const cols = boardColumns
    .sort((a, b) => a.position - b.position)
    .map((col) => ({
      id: col.id,
      name: col.name,
      tickets: [] as TicketData[],
    }));

  for (const ticket of tickets) {
    const col = cols.find((c) => c.id === ticket.column_id);
    if (col) {
      col.tickets.push(apiTicketToCard(ticket, memberIndex));
    }
  }

  for (const col of cols) {
    const posMap = new Map(
      tickets
        .filter((t) => t.column_id === col.id)
        .map((t) => [t.id, t.position]),
    );
    col.tickets.sort(
      (a, b) => (posMap.get(a.id) ?? 0) - (posMap.get(b.id) ?? 0),
    );
  }

  return cols;
}

export function BoardView({
  projectSlug,
  boardColumns,
  members,
  ticketFilter,
  sizePointsMap,
  weeklyPointsPerMember,
}: BoardViewProps) {
  const { token } = useAuth();
  const [columns, setColumns] = useState<ColumnData[]>([]);
  const [rawTickets, setRawTickets] = useState<TicketResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTicketId, setSelectedTicketId] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const memberIndex = new Map(members.map((m, i) => [m.user_id, i]));

  const applyFilter = useCallback(
    (tickets: TicketResponse[]) =>
      ticketFilter ? tickets.filter(ticketFilter) : tickets,
    [ticketFilter],
  );

  useEffect(() => {
    async function load() {
      if (!token) return;
      const { data, error } = await getBoardTickets(token, projectSlug);
      if (error) {
        toast.error(error);
        setIsLoading(false);
        return;
      }
      if (data) {
        setRawTickets(data.tickets);
        setColumns(
          buildColumns(boardColumns, applyFilter(data.tickets), memberIndex),
        );

        if (data.tickets.some((ticket) => !ticket.size_bucket)) {
          const sized = await classifyMissingTicketSizes(token, projectSlug);
          if (sized.data && sized.data.updated_count > 0) {
            setRawTickets(sized.data.tickets);
            setColumns(
              buildColumns(
                boardColumns,
                applyFilter(sized.data.tickets),
                memberIndex,
              ),
            );

            toast.success(
              <div className="flex items-center gap-2">
                <Sparkles className="size-4 text-primary" />
                <span>AI automatically sized {sized.data.updated_count} ticket{sized.data.updated_count === 1 ? "" : "s"}</span>
              </div>
            );
          }
        }
      }
      setIsLoading(false);
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, projectSlug]);

  // Re-filter when filter criteria change
  useEffect(() => {
    if (rawTickets.length > 0) {
      setColumns(
        buildColumns(boardColumns, applyFilter(rawTickets), memberIndex),
      );
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticketFilter]);

  const selectedRawTicket = selectedTicketId
    ? (rawTickets.find((t) => t.id === selectedTicketId) ?? null)
    : null;

  const handleTicketClick = useCallback((ticketId: string) => {
    setSelectedTicketId(ticketId);
    setModalOpen(true);
  }, []);

  const onDragEnd = useCallback(
    async (result: DropResult) => {
      const { source, destination, draggableId } = result;
      if (!destination) return;
      if (
        source.droppableId === destination.droppableId &&
        source.index === destination.index
      )
        return;

      setColumns((prev) => {
        const updated = prev.map((col) => ({
          ...col,
          tickets: [...col.tickets],
        }));
        const sourceCol = updated.find((c) => c.id === source.droppableId);
        const destCol = updated.find((c) => c.id === destination.droppableId);
        if (!sourceCol || !destCol) return prev;
        const [moved] = sourceCol.tickets.splice(source.index, 1);
        destCol.tickets.splice(destination.index, 0, moved);
        return updated;
      });

      const ticket = rawTickets.find((t) => t.id === draggableId);
      if (!ticket || !token) return;

      const { error } = await apiMoveTicket(
        token,
        projectSlug,
        ticket.ticket_key,
        {
          column_id: destination.droppableId,
          position: destination.index,
        },
      );

      if (error) {
        toast.error("Failed to move ticket");
        const { data } = await getBoardTickets(token, projectSlug);
        if (data) {
          setRawTickets(data.tickets);
          setColumns(
            buildColumns(boardColumns, applyFilter(data.tickets), memberIndex),
          );
        }
      } else {
        setRawTickets((prev) =>
          prev.map((t) =>
            t.id === draggableId
              ? {
                  ...t,
                  column_id: destination.droppableId,
                  position: destination.index,
                }
              : t,
          ),
        );
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [token, projectSlug, rawTickets],
  );

  const handleCreateTicket = useCallback(
    async (columnId: string, title: string) => {
      if (!token) return;
      const { data, error } = await apiCreateTicket(token, projectSlug, {
        title,
        column_id: columnId,
      });
      if (error) {
        toast.error(error);
        return;
      }
      if (data) {
        setRawTickets((prev) => [...prev, data]);
        setColumns((prev) =>
          prev.map((col) =>
            col.id === columnId
              ? {
                  ...col,
                  tickets: [...col.tickets, apiTicketToCard(data, memberIndex)],
                }
              : col,
          ),
        );
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [token, projectSlug],
  );

  const handleTicketUpdated = useCallback(
    (updated: TicketResponse) => {
      setRawTickets((prev) =>
        prev.map((t) => (t.id === updated.id ? updated : t)),
      );
      setColumns((prev) =>
        prev.map((col) => ({
          ...col,
          tickets: col.tickets.map((t) =>
            t.id === updated.id ? apiTicketToCard(updated, memberIndex) : t,
          ),
        })),
      );
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const handleTicketDeleted = useCallback(
    (ticketKey: string) => {
      const ticket = rawTickets.find((t) => t.ticket_key === ticketKey);
      if (!ticket) return;
      setRawTickets((prev) => prev.filter((t) => t.ticket_key !== ticketKey));
      setColumns((prev) =>
        prev.map((col) => ({
          ...col,
          tickets: col.tickets.filter((t) => t.id !== ticket.id),
        })),
      );
    },
    [rawTickets],
  );

  // Premium Skeleton Loading State
  if (isLoading) {
    return (
      <div className="flex h-full items-start gap-4 overflow-x-auto overflow-y-hidden px-2 pb-4 pt-2 opacity-60">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="flex h-full w-[340px] shrink-0 flex-col rounded-xl bg-muted/20 border border-border/40 p-3">
            <div className="mb-4 flex items-center justify-between px-1">
              <div className="h-4 w-28 rounded bg-muted/60 animate-pulse" />
              <div className="h-4 w-6 rounded bg-muted/60 animate-pulse" />
            </div>
            <div className="flex flex-col gap-3">
              {[1, 2].map((j) => (
                <div key={j} className="h-28 w-full rounded-lg bg-background shadow-sm border border-border/40 animate-pulse" />
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <>
      <DragDropContext onDragEnd={onDragEnd}>
        {/* Adjusted flex container to properly handle overflow and padding */}
        <div className="flex h-full items-start gap-4 overflow-x-auto overflow-y-hidden px-2 pb-4 pt-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-muted-foreground/20 hover:scrollbar-thumb-muted-foreground/40">
          {columns.map((column, idx) => (
            <BoardColumn
              key={column.id}
              column={column}
              isLast={idx === columns.length - 1}
              onCreateTicket={handleCreateTicket}
              onTicketClick={handleTicketClick}
            />
          ))}
          {/* Invisible spacer to ensure padding at the end of the scroll container */}
          <div className="w-2 shrink-0" />
        </div>
      </DragDropContext>

      <TicketDetailModal
        key={
          selectedRawTicket
            ? `${selectedRawTicket.ticket_key}:${modalOpen ? "open" : "closed"}`
            : "ticket-modal"
        }
        ticket={selectedRawTicket}
        projectSlug={projectSlug}
        members={members}
        sizePointsMap={sizePointsMap}
        weeklyPointsPerMember={weeklyPointsPerMember}
        open={modalOpen}
        onClose={() => {
          setModalOpen(false);
          setSelectedTicketId(null);
        }}
        onUpdated={handleTicketUpdated}
        onDeleted={handleTicketDeleted}
      />
    </>
  );
}
