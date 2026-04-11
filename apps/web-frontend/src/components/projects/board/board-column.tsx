"use client";

import { Droppable } from "@hello-pangea/dnd";
import { Sparkles } from "lucide-react";

import { cn } from "@/lib/utils";
import { BoardCard, type TicketData } from "./board-card";
import { CreateTicketInline } from "./create-ticket-inline";

export interface ColumnData {
  id: string;
  name: string;
  tickets: TicketData[];
}

interface BoardColumnProps {
  column: ColumnData;
  isLast: boolean;
  onCreateTicket: (columnId: string, title: string) => void;
  onTicketClick: (ticketId: string) => void;
}

export function BoardColumn({
  column,
  isLast,
  onCreateTicket,
  onTicketClick,
}: BoardColumnProps) {
  return (
    <div className="flex h-full w-[340px] shrink-0 flex-col rounded-xl border border-border/40 bg-muted/10 shadow-sm overflow-hidden">
      
      {/* ========== COLUMN HEADER ========== */}
      <div className="flex shrink-0 items-center justify-between px-3.5 py-3 border-b border-border/40 bg-muted/20">
        <div className="flex items-center gap-2.5">
          <h3 className="text-[11px] font-bold uppercase tracking-wider text-foreground/80">
            {column.name}
          </h3>
          <div className="flex h-5 min-w-[20px] items-center justify-center rounded-full bg-background px-1.5 text-[10px] font-bold text-muted-foreground shadow-sm ring-1 ring-border/50">
            {column.tickets.length}
          </div>
        </div>
        
        {isLast && (
          <div className="flex items-center gap-1 rounded bg-primary/10 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider text-primary ring-1 ring-primary/20">
            <Sparkles className="size-2.5" />
            Done
          </div>
        )}
      </div>

      {/* ========== DROPPABLE CARD AREA ========== */}
      <Droppable droppableId={column.id}>
        {(provided, snapshot) => (
          <div
            ref={provided.innerRef}
            {...provided.droppableProps}
            className={cn(
              "flex min-h-[150px] flex-1 flex-col gap-2.5 overflow-y-auto p-2.5 transition-colors",
              // Custom scrollbar styling
              "scrollbar-thin scrollbar-track-transparent scrollbar-thumb-muted-foreground/20 hover:scrollbar-thumb-muted-foreground/40",
              // Drag over visual state
              snapshot.isDraggingOver
                ? "bg-primary/[0.03] ring-1 ring-inset ring-primary/20"
                : "bg-transparent"
            )}
          >
            {column.tickets.map((ticket, idx) => (
              <BoardCard
                key={ticket.id}
                ticket={ticket}
                index={idx}
                onClick={onTicketClick}
              />
            ))}
            
            {provided.placeholder}

            {/* Empty State */}
            {column.tickets.length === 0 && !snapshot.isDraggingOver && (
              <div className="mx-1 mt-1 flex flex-1 flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-border/60 bg-background/50 py-10 transition-colors">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/50">
                  Drop tickets here
                </p>
              </div>
            )}
          </div>
        )}
      </Droppable>

      {/* ========== CREATE TICKET FOOTER ========== */}
      <div className="shrink-0 border-t border-border/40 bg-muted/5 p-2">
        <CreateTicketInline
          columnId={column.id}
          onCreateTicket={onCreateTicket}
        />
      </div>
      
    </div>
  );
}