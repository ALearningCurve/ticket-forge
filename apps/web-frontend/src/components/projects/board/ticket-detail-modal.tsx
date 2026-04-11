"use client";

import { useState, useCallback, useEffect } from "react";
import {
  AlertTriangle,
  Bookmark,
  Calendar as CalendarIcon,
  CheckSquare,
  Clock,
  Loader2,
  Sparkles,
  Tag,
  Target,
  Trash2,
  User,
  X,
} from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import { useAuth } from "@/lib/auth-context";
import {
  getTicketEngineerRecommendations,
  updateTicket,
  deleteTicket as apiDeleteTicket,
  type TicketResponse,
  type ProjectMember,
  type RecommendedEngineerResponse,
} from "@/lib/api";
import { cn } from "@/lib/utils";

interface TicketDetailModalProps {
  ticket: TicketResponse | null;
  projectSlug: string;
  members: ProjectMember[];
  sizePointsMap?: { S: number; M: number; L: number; XL: number };
  weeklyPointsPerMember?: number;
  open: boolean;
  onClose: () => void;
  onUpdated: (ticket: TicketResponse) => void;
  onDeleted: (ticketKey: string) => void;
}

const priorityOptions = [
  {
    value: "critical",
    label: "Critical",
    color: "bg-red-500",
    ring: "ring-red-500/20",
  },
  {
    value: "high",
    label: "High",
    color: "bg-orange-500",
    ring: "ring-orange-500/20",
  },
  {
    value: "medium",
    label: "Medium",
    color: "bg-yellow-500",
    ring: "ring-yellow-500/20",
  },
  {
    value: "low",
    label: "Low",
    color: "bg-blue-400",
    ring: "ring-blue-400/20",
  },
];

const typeOptions = [
  {
    value: "task",
    label: "Task",
    icon: CheckSquare,
    color: "text-blue-500",
    bg: "bg-blue-50 dark:bg-blue-950/30",
  },
  {
    value: "story",
    label: "Story",
    icon: Bookmark,
    color: "text-green-500",
    bg: "bg-green-50 dark:bg-green-950/30",
  },
  {
    value: "bug",
    label: "Bug",
    icon: AlertTriangle,
    color: "text-red-500",
    bg: "bg-red-50 dark:bg-red-950/30",
  },
];

const sizeOptions = [
  { value: "auto", label: "Auto (AI estimate)", short: "AI" },
  { value: "S", label: "Small", short: "S" },
  { value: "M", label: "Medium", short: "M" },
  { value: "L", label: "Large", short: "L" },
  { value: "XL", label: "Extra Large", short: "XL" },
];

const sizeColors: Record<string, string> = {
  S: "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300",
  M: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  L: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  XL: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
};

const COMMON_LABELS = [
  "frontend",
  "backend",
  "design",
  "docs",
  "infrastructure",
  "testing",
  "security",
  "performance",
];

const labelColors: Record<string, string> = {
  frontend:
    "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300 border-purple-200 dark:border-purple-800",
  backend:
    "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 border-blue-200 dark:border-blue-800",
  design:
    "bg-pink-100 text-pink-700 dark:bg-pink-950 dark:text-pink-300 border-pink-200 dark:border-pink-800",
  docs: "bg-teal-100 text-teal-700 dark:bg-teal-950 dark:text-teal-300 border-teal-200 dark:border-teal-800",
  infrastructure:
    "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300 border-amber-200 dark:border-amber-800",
  testing:
    "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 border-green-200 dark:border-green-800",
  security:
    "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300 border-orange-200 dark:border-orange-800",
  performance:
    "bg-cyan-100 text-cyan-700 dark:bg-cyan-950 dark:text-cyan-300 border-cyan-200 dark:border-cyan-800",
};

function getLabelColor(label: string) {
  return (
    labelColors[label.toLowerCase()] ||
    "bg-muted text-muted-foreground border-border"
  );
}

function getDaysUntilDue(dateStr: string): {
  text: string;
  urgent: boolean;
  overdue: boolean;
} {
  const due = new Date(dateStr + "T00:00:00");
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diff = Math.ceil(
    (due.getTime() - today.getTime()) / (1000 * 60 * 60 * 24),
  );

  if (diff < 0)
    return { text: `${Math.abs(diff)}d overdue`, urgent: true, overdue: true };
  if (diff === 0) return { text: "Due today", urgent: true, overdue: false };
  if (diff === 1) return { text: "Due tomorrow", urgent: true, overdue: false };
  if (diff <= 3)
    return { text: `${diff} days left`, urgent: true, overdue: false };
  return { text: `${diff} days left`, urgent: false, overdue: false };
}

export function TicketDetailModal({
  ticket,
  projectSlug,
  members,
  sizePointsMap = { S: 1, M: 2, L: 3, XL: 5 },
  weeklyPointsPerMember = 10,
  open,
  onClose,
  onUpdated,
  onDeleted,
}: TicketDetailModalProps) {
  type SizeValue = "S" | "M" | "L" | "XL";
  type SizeMode = SizeValue | "auto";

  const { token } = useAuth();
  const [title, setTitle] = useState(ticket?.title || "");
  const [description, setDescription] = useState(ticket?.description || "");
  const [priority, setPriority] = useState(ticket?.priority || "medium");
  const [type, setType] = useState(ticket?.type || "task");
  const [sizeMode, setSizeMode] = useState<SizeMode>(
    ticket?.size_source === "manual" && ticket?.size_bucket
      ? (ticket.size_bucket as SizeValue)
      : "auto",
  );
  const [assigneeId, setAssigneeId] = useState<string | null>(
    ticket?.assignee?.id || null,
  );
  const [dueDate, setDueDate] = useState(ticket?.due_date || "");
  const [labels, setLabels] = useState<string[]>(ticket?.labels || []);
  const [newLabel, setNewLabel] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showLabelInput, setShowLabelInput] = useState(false);

  // Real API recommendations
  const [recommendations, setRecommendations] = useState<
    RecommendedEngineerResponse[]
  >([]);
  const [recError, setRecError] = useState<string | null>(null);
  const [recLoading, setRecLoading] = useState(false);

  useEffect(() => {
    async function loadRecommendations() {
      if (!open || !token || !ticket) return;
      setRecLoading(true);
      setRecError(null);
      const { data, error } = await getTicketEngineerRecommendations(
        token,
        projectSlug,
        ticket.ticket_key,
      );
      if (error) {
        setRecommendations([]);
        setRecError(error);
      } else {
        setRecommendations(data?.recommendations ?? []);
      }
      setRecLoading(false);
    }
    void loadRecommendations();
  }, [open, projectSlug, ticket?.ticket_key, token]);

  const handleSave = useCallback(async () => {
    if (!token || !ticket) return;
    setIsSaving(true);

    const { data, error } = await updateTicket(
      token,
      projectSlug,
      ticket.ticket_key,
      {
        title: title.trim(),
        description: description.trim() || undefined,
        priority,
        type,
        size_bucket: sizeMode === "auto" ? null : sizeMode,
        assignee_id: assigneeId,
        due_date: dueDate || undefined,
        labels,
      },
    );

    setIsSaving(false);
    if (error) {
      toast.error(error);
      return;
    }
    if (data) {
      toast.success("Ticket updated");
      onUpdated(data);
      onClose();
    }
  }, [
    token,
    ticket,
    projectSlug,
    title,
    description,
    priority,
    type,
    sizeMode,
    assigneeId,
    dueDate,
    labels,
    onUpdated,
    onClose,
  ]);

  const handleDelete = useCallback(async () => {
    if (!token || !ticket) return;
    const confirmed = window.confirm(
      `Delete ${ticket.ticket_key}? This cannot be undone.`,
    );
    if (!confirmed) return;

    setIsDeleting(true);
    const { error } = await apiDeleteTicket(
      token,
      projectSlug,
      ticket.ticket_key,
    );
    setIsDeleting(false);
    if (error) {
      toast.error(error);
      return;
    }
    toast.success("Ticket deleted");
    onDeleted(ticket.ticket_key);
    onClose();
  }, [token, ticket, projectSlug, onDeleted, onClose]);

  function addLabel(label: string) {
    const trimmed = label.trim().toLowerCase();
    if (trimmed && !labels.includes(trimmed)) {
      setLabels([...labels, trimmed]);
    }
    setNewLabel("");
    setShowLabelInput(false);
  }

  function removeLabel(label: string) {
    setLabels(labels.filter((l) => l !== label));
  }

  if (!ticket) return null;

  const currentType = typeOptions.find((t) => t.value === type);
  const TypeIcon = currentType?.icon || CheckSquare;
  const currentPriority = priorityOptions.find((p) => p.value === priority);
  const selectedOrPredictedSize = (
    sizeMode === "auto" ? ticket.size_bucket || ticket.size || "M" : sizeMode
  ) as SizeValue;
  const pointsCost =
    sizePointsMap[selectedOrPredictedSize as keyof typeof sizePointsMap] || 0;
  const dueInfo = dueDate ? getDaysUntilDue(dueDate) : null;

  // Top recommendation for the suggested assignee slot
  const topRec = recommendations.length > 0 ? recommendations[0] : null;

  return (
    <Dialog key={ticket.id} open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="flex max-h-[90vh] w-[94vw] flex-col overflow-hidden p-0 sm:w-[55vw] sm:!max-w-[900px]">
        {/* ========== HEADER ========== */}
        <DialogHeader className="shrink-0 border-b bg-background px-5 py-3.5 shadow-sm z-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-2 py-1",
                  currentType?.bg,
                )}
              >
                <TypeIcon className={cn("size-4", currentType?.color)} />
                <span className="text-xs font-semibold">
                  {currentType?.label}
                </span>
              </div>
              <span className="text-sm font-mono font-medium text-muted-foreground/80">
                {ticket.ticket_key}
              </span>
            </div>
            <div className="flex items-center gap-2.5">
              <div className="flex items-center gap-1.5 rounded-md bg-muted/60 px-2.5 py-1">
                <Target className="size-3.5 text-muted-foreground" />
                <span className="text-xs font-semibold">
                  {pointsCost} {pointsCost === 1 ? "pt" : "pts"}
                </span>
              </div>
              {dueInfo && (
                <div
                  className={cn(
                    "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-semibold",
                    dueInfo.overdue
                      ? "bg-red-100 text-red-700 dark:bg-red-950/50 dark:text-red-400"
                      : dueInfo.urgent
                        ? "bg-amber-100 text-amber-700 dark:bg-amber-950/50 dark:text-amber-400"
                        : "bg-muted/60 text-muted-foreground",
                  )}
                >
                  <Clock className="size-3.5" />
                  {dueInfo.text}
                </div>
              )}
            </div>
          </div>
          <DialogTitle className="sr-only">
            Edit {ticket.ticket_key}
          </DialogTitle>
        </DialogHeader>

        {/* ========== SCROLLABLE BODY ========== */}
        <div className="flex flex-1 flex-col overflow-hidden md:flex-row">
          {/* LEFT SIDE: Main content */}
          <div className="flex-1 overflow-y-auto px-6 py-5">
            <div className="space-y-6">
              {/* Title */}
              <div>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="h-auto border-none px-0 py-1 text-2xl font-semibold tracking-tight shadow-none focus-visible:ring-0 placeholder:text-muted-foreground/50"
                  placeholder="Ticket title"
                />
              </div>

              {/* Description */}
              <div className="space-y-2">
                <Label className="text-xs font-semibold text-muted-foreground">
                  Description
                </Label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Add a detailed description..."
                  className="min-h-[140px] w-full resize-y rounded-lg border bg-muted/20 px-3.5 py-3 text-sm placeholder:text-muted-foreground/40 focus:bg-background focus:outline-none focus:ring-2 focus:ring-ring/50 transition-colors"
                />
              </div>

              {/* Assignee Split */}
              <div className="space-y-2">
                <Label className="text-xs font-semibold text-muted-foreground">
                  Assignee
                </Label>
                <div className="grid gap-4 sm:grid-cols-2 items-start">
                  {/* Manual Assignment */}
                  <div className="space-y-2">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/50">
                      Select Member
                    </p>
                    <Select
                      value={assigneeId || "unassigned"}
                      onValueChange={(v) =>
                        setAssigneeId(v === "unassigned" ? null : v)
                      }
                    >
                      <SelectTrigger className="w-full h-10">
                        <SelectValue placeholder="Unassigned" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="unassigned">
                          <div className="flex items-center gap-2.5">
                            <div className="flex size-5 items-center justify-center rounded-full border border-dashed">
                              <User className="size-3 text-muted-foreground" />
                            </div>
                            <span>Unassigned</span>
                          </div>
                        </SelectItem>
                        {members.map((member) => (
                          <SelectItem
                            key={member.user_id}
                            value={member.user_id}
                          >
                            <div className="flex items-center gap-2.5">
                              <div className="flex size-5 items-center justify-center rounded-full bg-primary text-[9px] font-bold text-primary-foreground">
                                {member.first_name[0]}
                                {member.last_name[0]}
                              </div>
                              <span>
                                {member.first_name} {member.last_name}
                              </span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* AI Recommendation — top pick */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-1">
                      <Sparkles className="size-3.5 text-primary" />
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-primary">
                        Suggested Assignee
                      </p>
                    </div>
                    {recLoading ? (
                      <div className="flex h-10 items-center justify-center gap-2 rounded-lg border border-dashed text-xs text-muted-foreground">
                        <Loader2 className="size-3 animate-spin" />
                        Loading...
                      </div>
                    ) : topRec ? (
                      <button
                        type="button"
                        onClick={() => setAssigneeId(topRec.user_id)}
                        className="group flex w-full items-center gap-3 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-left transition-all hover:border-primary/40 hover:bg-primary/10 h-10"
                      >
                        <div className="flex size-6 shrink-0 items-center justify-center rounded-full bg-primary text-[10px] font-bold text-primary-foreground">
                          {topRec.first_name[0]}
                          {topRec.last_name[0]}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-medium leading-none">
                            {topRec.first_name} {topRec.last_name}
                          </p>
                        </div>
                        <div className="flex items-center gap-1 text-primary">
                          <span className="text-xs font-bold">
                            {Math.round(topRec.recommendation_score * 100)}%
                          </span>
                        </div>
                      </button>
                    ) : (
                      <div className="flex h-10 items-center justify-center rounded-lg border border-dashed text-xs text-muted-foreground bg-muted/10">
                        No recommendation
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Labels */}
              <div className="space-y-3">
                <Label className="text-xs font-semibold text-muted-foreground">
                  Labels
                </Label>
                <div className="flex flex-wrap items-center gap-2">
                  {labels.map((label) => (
                    <span
                      key={label}
                      className={cn(
                        "inline-flex items-center gap-1 rounded-md border px-2.5 py-1 text-xs font-medium",
                        getLabelColor(label),
                      )}
                    >
                      {label}
                      <button
                        type="button"
                        onClick={() => removeLabel(label)}
                        className="rounded-full p-0.5 opacity-60 hover:opacity-100 transition-opacity"
                      >
                        <X className="size-3" />
                      </button>
                    </span>
                  ))}

                  {showLabelInput ? (
                    <Input
                      value={newLabel}
                      onChange={(e) => setNewLabel(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") addLabel(newLabel);
                        if (e.key === "Escape") {
                          setShowLabelInput(false);
                          setNewLabel("");
                        }
                      }}
                      placeholder="Type & enter"
                      className="h-7 w-32 text-xs"
                      autoFocus
                    />
                  ) : (
                    <button
                      onClick={() => setShowLabelInput(true)}
                      className="inline-flex h-7 items-center gap-1.5 rounded-md border border-dashed px-3 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
                    >
                      <Tag className="size-3" />
                      Add label
                    </button>
                  )}
                </div>

                {showLabelInput && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {COMMON_LABELS.filter((l) => !labels.includes(l))
                      .slice(0, 6)
                      .map((label) => (
                        <button
                          key={label}
                          onClick={() => addLabel(label)}
                          className={cn(
                            "rounded-md border px-2 py-0.5 text-[11px] font-medium transition-colors hover:opacity-80",
                            getLabelColor(label),
                          )}
                        >
                          {label}
                        </button>
                      ))}
                  </div>
                )}
              </div>

              {/* Recommended Engineers — full list */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="size-3.5 text-primary" />
                  <Label className="text-xs font-semibold text-primary">
                    Recommended engineers
                  </Label>
                </div>

                {recLoading ? (
                  <div className="flex items-center gap-2 rounded-lg border bg-muted/20 p-4 text-xs text-muted-foreground">
                    <Loader2 className="size-3.5 animate-spin" />
                    Loading recommendations...
                  </div>
                ) : recError ? (
                  <p className="rounded-lg border bg-muted/20 p-4 text-xs text-muted-foreground">
                    {recError}
                  </p>
                ) : recommendations.length === 0 ? (
                  <p className="rounded-lg border bg-muted/20 p-4 text-xs text-muted-foreground">
                    No engineer recommendations available yet.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {recommendations.map((rec) => (
                      <button
                        key={rec.user_id}
                        type="button"
                        onClick={() => setAssigneeId(rec.user_id)}
                        className="w-full rounded-md border bg-background p-3 text-left transition-colors hover:bg-muted/50"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="text-sm font-medium">
                              {rec.first_name} {rec.last_name}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              @{rec.username}
                            </p>
                          </div>
                          <Badge
                            variant={rec.has_capacity ? "default" : "secondary"}
                            className="text-[10px]"
                          >
                            {rec.has_capacity ? "Has capacity" : "Busy"}
                          </Badge>
                        </div>
                        <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-muted-foreground">
                          <p>
                            Match: {Math.round(rec.recommendation_score * 100)}%
                          </p>
                          <p>Active tickets: {rec.active_ticket_count}</p>
                          <p>
                            Semantic:{" "}
                            {Math.round(rec.semantic_similarity * 100)}%
                          </p>
                          <p>
                            Capacity: {Math.round(rec.capacity_score * 100)}%
                          </p>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* RIGHT SIDEBAR: Metadata */}
          <div className="w-full shrink-0 border-t md:border-l md:border-t-0 bg-muted/10 p-5 md:w-[260px] md:overflow-y-auto flex flex-col gap-5">
            <div className="grid grid-cols-2 gap-4 md:grid-cols-1 md:gap-5">
              {/* Priority */}
              <div className="space-y-1.5">
                <Label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                  Priority
                </Label>
                <Select value={priority} onValueChange={setPriority}>
                  <SelectTrigger className="h-9 text-sm bg-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {priorityOptions.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        <div className="flex items-center gap-2">
                          <span
                            className={cn("size-2.5 rounded-full", opt.color)}
                          />
                          <span>{opt.label}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Size */}
              <div className="space-y-1.5">
                <Label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                  Size
                </Label>
                <Select
                  value={sizeMode}
                  onValueChange={(value) => setSizeMode(value as SizeMode)}
                >
                  <SelectTrigger className="h-9 text-sm bg-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {sizeOptions.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        <div className="flex items-center gap-2">
                          {opt.value === "auto" ? (
                            <span className="flex size-5 items-center justify-center rounded bg-muted text-[10px] font-bold text-muted-foreground">
                              {opt.short}
                            </span>
                          ) : (
                            <span
                              className={cn(
                                "flex size-5 items-center justify-center rounded text-[10px] font-bold",
                                sizeColors[opt.value],
                              )}
                            >
                              {opt.short}
                            </span>
                          )}
                          <span>{opt.label}</span>
                          {opt.value !== "auto" && (
                            <span className="ml-auto text-[10px] text-muted-foreground">
                              (
                              {
                                sizePointsMap[
                                  opt.value as keyof typeof sizePointsMap
                                ]
                              }
                              pt)
                            </span>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Type */}
              <div className="space-y-1.5">
                <Label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                  Type
                </Label>
                <Select value={type} onValueChange={setType}>
                  <SelectTrigger className="h-9 text-sm bg-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {typeOptions.map((opt) => {
                      const Icon = opt.icon;
                      return (
                        <SelectItem key={opt.value} value={opt.value}>
                          <div className="flex items-center gap-2">
                            <Icon className={cn("size-3.5", opt.color)} />
                            <span>{opt.label}</span>
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectContent>
                </Select>
              </div>

              {/* Due date */}
              <div className="space-y-1.5">
                <Label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                  Due date
                </Label>
                <div className="relative">
                  <CalendarIcon className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    type="date"
                    value={dueDate}
                    onChange={(e) => setDueDate(e.target.value)}
                    className="h-9 pl-9 text-sm bg-background"
                  />
                </div>
                {dueDate && (
                  <div className="flex justify-end pt-1">
                    <button
                      onClick={() => setDueDate("")}
                      className="text-[10px] font-medium text-muted-foreground hover:text-foreground"
                    >
                      Clear date
                    </button>
                  </div>
                )}
              </div>
            </div>

            <Separator className="hidden md:block" />

            {/* Quick Summary */}
            <div className="rounded-lg border bg-background/50 p-3 space-y-2">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70 mb-2">
                Details
              </p>
              <div className="grid grid-cols-2 gap-y-2.5 text-xs">
                <span className="text-muted-foreground">Sprint Load</span>
                <span className="font-semibold text-right">
                  {pointsCost} / {weeklyPointsPerMember}
                </span>
                <span className="text-muted-foreground">Status</span>
                <span className="text-right">
                  <span
                    className={cn(
                      "rounded px-1.5 py-0.5 text-[10px] font-bold",
                      sizeColors[selectedOrPredictedSize],
                    )}
                  >
                    {sizeMode === "auto"
                      ? `AI ${selectedOrPredictedSize}`
                      : selectedOrPredictedSize}
                  </span>
                </span>
                <span className="text-muted-foreground">Level</span>
                <span className="flex items-center justify-end gap-1.5">
                  <span
                    className={cn(
                      "size-2 rounded-full",
                      currentPriority?.color,
                    )}
                  />
                  <span className="capitalize font-medium">{priority}</span>
                </span>
              </div>
            </div>

            <div className="mt-auto pt-4 space-y-2 text-[11px] text-muted-foreground">
              <div className="flex items-center justify-between">
                <span>Created</span>
                <span className="font-medium text-foreground/80">
                  {new Date(ticket.created_at).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Updated</span>
                <span className="font-medium text-foreground/80">
                  {new Date(ticket.updated_at).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                    year: "numeric",
                  })}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ========== FOOTER ========== */}
        <div className="shrink-0 flex items-center justify-between border-t bg-background px-5 py-3 shadow-sm z-10">
          <Button
            variant="ghost"
            size="sm"
            className="text-destructive hover:bg-destructive/10 hover:text-destructive h-9 px-3"
            onClick={handleDelete}
            disabled={isDeleting}
          >
            {isDeleting ? (
              <Loader2 className="mr-2 size-4 animate-spin" />
            ) : (
              <Trash2 className="mr-2 size-4" />
            )}
            Delete Ticket
          </Button>
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={onClose} className="h-9">
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={isSaving || !title.trim()}
              className="h-9 px-4 font-medium"
            >
              {isSaving && <Loader2 className="mr-2 size-4 animate-spin" />}
              Save changes
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
