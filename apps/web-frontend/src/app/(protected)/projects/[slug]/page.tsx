"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import { Filter, Loader2, Search, Settings, X, ChevronRight } from "lucide-react";
import { toast } from "sonner";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { BoardView } from "@/components/projects/board/board-view";
import { TeamView } from "@/components/projects/team-view";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";
import {
  getProject,
  getBoardTickets,
  type ProjectResponse,
  type TicketResponse,
} from "@/lib/api";

type TabType = "board" | "team";

const PRIORITY_OPTIONS = ["critical", "high", "medium", "low"];
const TYPE_OPTIONS = ["task", "story", "bug"];

const priorityColors: Record<string, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-yellow-500",
  low: "bg-blue-400",
};

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { token, user } = useAuth();
  const slug = params.slug as string;

  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [tickets, setTickets] = useState<TicketResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabType>("board");

  // Filter state
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMemberIds, setSelectedMemberIds] = useState<Set<string>>(
    new Set(),
  );
  const [selectedPriorities, setSelectedPriorities] = useState<Set<string>>(
    new Set(),
  );
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [selectedLabels, setSelectedLabels] = useState<Set<string>>(new Set());

  // Initial load
  useEffect(() => {
    async function load() {
      if (!token) return;

      const [projectRes, ticketsRes] = await Promise.all([
        getProject(token, slug),
        getBoardTickets(token, slug),
      ]);

      if (projectRes.error) {
        toast.error(projectRes.error);
        router.push("/dashboard");
        return;
      }
      if (projectRes.data) setProject(projectRes.data);
      if (ticketsRes.data) setTickets(ticketsRes.data.tickets);

      setIsLoading(false);
    }
    load();
  }, [token, slug, router]);

  const refreshTickets = useCallback(async () => {
    if (!token) return;
    const { data } = await getBoardTickets(token, slug);
    if (data) setTickets(data.tickets);
  }, [token, slug]);

  function handleTabChange(tab: TabType) {
    setActiveTab(tab);
    if (tab === "team") {
      refreshTickets();
    }
  }

  // Toggle helpers
  function toggleMember(userId: string) {
    setSelectedMemberIds((prev) => {
      const next = new Set(prev);
      if (next.has(userId)) {
        next.delete(userId);
      } else {
        next.add(userId);
      }
      return next;
    });
  }

  function toggleFilter(
    set: Set<string>,
    setter: (s: Set<string>) => void,
    value: string,
  ) {
    const next = new Set(set);
    if (next.has(value)) {
      next.delete(value);
    } else {
      next.add(value);
    }
    setter(next);
  }

  function clearAllFilters() {
    setSearchQuery("");
    setSelectedMemberIds(new Set());
    setSelectedPriorities(new Set());
    setSelectedTypes(new Set());
    setSelectedLabels(new Set());
  }

  // Collect all unique labels from tickets
  const allLabels = useMemo(() => {
    const labelSet = new Set<string>();
    tickets.forEach((t) => t.labels?.forEach((l) => labelSet.add(l)));
    return Array.from(labelSet).sort();
  }, [tickets]);

  // Filter tickets
  const filteredTickets = useMemo(() => {
    return tickets.filter((ticket) => {
      // Search
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const matchesSearch =
          ticket.title.toLowerCase().includes(q) ||
          ticket.ticket_key.toLowerCase().includes(q) ||
          (ticket.description || "").toLowerCase().includes(q);
        if (!matchesSearch) return false;
      }

      // Member filter
      if (selectedMemberIds.size > 0) {
        if (!ticket.assignee || !selectedMemberIds.has(ticket.assignee.id)) {
          return false;
        }
      }

      // Priority filter
      if (selectedPriorities.size > 0) {
        if (!selectedPriorities.has(ticket.priority)) return false;
      }

      // Type filter
      if (selectedTypes.size > 0) {
        if (!selectedTypes.has(ticket.type)) return false;
      }

      // Label filter
      if (selectedLabels.size > 0) {
        const ticketLabels = new Set(ticket.labels || []);
        const hasMatchingLabel = [...selectedLabels].some((l) =>
          ticketLabels.has(l),
        );
        if (!hasMatchingLabel) return false;
      }

      return true;
    });
  }, [
    tickets,
    searchQuery,
    selectedMemberIds,
    selectedPriorities,
    selectedTypes,
    selectedLabels,
  ]);

  const activeFilterCount =
    selectedPriorities.size + selectedTypes.size + selectedLabels.size;
  const hasAnyFilter =
    searchQuery.length > 0 ||
    selectedMemberIds.size > 0 ||
    activeFilterCount > 0;

  const myRole = project?.members.find((m) => m.user_id === user?.id)?.role;
  const canManage = myRole === "owner" || myRole === "admin";

  if (isLoading) {
    return (
      <div className="flex h-full min-h-[calc(100vh-4rem)] w-full items-center justify-center bg-background">
        <Loader2 className="size-6 animate-spin text-primary" />
      </div>
    );
  }

  if (!project) return null;

  const memberColors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"];

  const tabs: { key: TabType; label: string }[] = [
    { key: "board", label: "Board" },
    { key: "team", label: "Team" },
  ];

  return (
    <div className="flex h-[calc(100vh-4rem)] w-full flex-1 min-w-0 flex-col overflow-hidden bg-background">
      
      {/* ========== HEADER ========== */}
      <div className="shrink-0 border-b bg-background px-4 sm:px-6 lg:px-8 pt-5 w-full">
        <div className="mb-5 flex flex-col gap-1.5">
          <div className="flex items-center gap-1.5 text-[13px] font-medium text-muted-foreground/80">
            <Link
              href="/dashboard"
              className="transition-colors hover:text-foreground"
            >
              Projects
            </Link>
            <ChevronRight className="size-3.5 opacity-50" />
            <span className="text-foreground">{project.name}</span>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">
            {project.name}
          </h1>
        </div>

        {/* Flush Tabs */}
        <div className="flex gap-6 w-full">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => handleTabChange(tab.key)}
              className={cn(
                "relative pb-3 text-sm font-medium transition-colors hover:text-foreground outline-none",
                activeTab === tab.key
                  ? "text-foreground"
                  : "text-muted-foreground"
              )}
            >
              {tab.label}
              {activeTab === tab.key && (
                <span className="absolute bottom-0 left-0 right-0 h-[2px] rounded-t-full bg-primary" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* ========== TOOLBAR ========== */}
      <div className="shrink-0 flex w-full items-center justify-between border-b bg-background px-4 sm:px-6 lg:px-8 py-3 z-10 shadow-sm">
        
        {/* Left Side: Board Filters vs Team Stats */}
        {activeTab === "board" ? (
          <div className="flex flex-1 flex-wrap items-center gap-3">
            {/* Search */}
            <div className="relative group flex-1 min-w-[200px] max-w-[240px] xl:max-w-[320px]">
              <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground transition-colors group-focus-within:text-foreground" />
              <Input
                placeholder="Search tickets..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-9 w-full pl-9 pr-8 text-sm bg-muted/30 border-border/60 transition-all focus-visible:bg-background focus-visible:ring-1 focus-visible:ring-primary/50 shadow-none"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-0.5 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors outline-none"
                >
                  <X className="size-3.5" />
                </button>
              )}
            </div>

            <Separator orientation="vertical" className="h-5 hidden sm:block mx-1" />

            {/* Member avatars */}
            <div className="flex items-center gap-3">
              <div className="flex -space-x-2">
                {project.members.slice(0, 5).map((member, idx) => {
                  const isSelected = selectedMemberIds.has(member.user_id);
                  return (
                    <button
                      key={member.id}
                      onClick={() => toggleMember(member.user_id)}
                      className={cn(
                        "relative flex size-8 items-center justify-center rounded-full text-[10px] font-bold text-white transition-all ring-2 ring-background outline-none hover:z-20 hover:scale-105",
                        isSelected
                          ? "ring-primary z-20 scale-105 shadow-md"
                          : selectedMemberIds.size > 0
                            ? "opacity-40 hover:opacity-100"
                            : ""
                      )}
                      style={{
                        backgroundColor: memberColors[idx % memberColors.length],
                      }}
                      title={`${member.first_name} ${member.last_name}`}
                    >
                      {member.first_name[0]}{member.last_name[0]}
                    </button>
                  );
                })}
                {project.members.length > 5 && (
                  <div className="flex size-8 items-center justify-center rounded-full bg-muted text-[10px] font-bold text-muted-foreground ring-2 ring-background z-0">
                    +{project.members.length - 5}
                  </div>
                )}
              </div>
            </div>

            <Separator orientation="vertical" className="h-5 hidden sm:block mx-1" />

            {/* Filter popover */}
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant={activeFilterCount > 0 ? "secondary" : "ghost"}
                  size="sm"
                  className={cn(
                    "h-9 text-[13px] font-medium outline-none",
                    activeFilterCount > 0 
                      ? "bg-secondary text-secondary-foreground" 
                      : "text-muted-foreground hover:text-foreground border border-dashed border-border/60 hover:bg-muted/50"
                  )}
                >
                  <Filter className="mr-2 size-3.5" />
                  Filters
                  {activeFilterCount > 0 && (
                    <Badge
                      variant="default"
                      className="ml-2 size-5 px-0 flex items-center justify-center rounded-full text-[10px] font-bold"
                    >
                      {activeFilterCount}
                    </Badge>
                  )}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[300px] p-4 shadow-xl rounded-xl" align="start">
                <div className="space-y-5">
                  {/* Priority */}
                  <div className="space-y-2.5">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70">
                      Priority
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {PRIORITY_OPTIONS.map((p) => (
                        <button
                          key={p}
                          onClick={() => toggleFilter(selectedPriorities, setSelectedPriorities, p)}
                          className={cn(
                            "flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs capitalize transition-colors font-medium outline-none",
                            selectedPriorities.has(p)
                              ? "border-primary bg-primary/10 text-primary"
                              : "border-border/50 hover:bg-muted text-muted-foreground hover:text-foreground"
                          )}
                        >
                          <span className={cn("size-2 rounded-full shadow-sm", priorityColors[p])} />
                          {p}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Type */}
                  <div className="space-y-2.5">
                    <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70">
                      Issue Type
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {TYPE_OPTIONS.map((t) => (
                        <button
                          key={t}
                          onClick={() => toggleFilter(selectedTypes, setSelectedTypes, t)}
                          className={cn(
                            "rounded-md border px-2.5 py-1.5 text-xs capitalize transition-colors font-medium outline-none",
                            selectedTypes.has(t)
                              ? "border-primary bg-primary/10 text-primary"
                              : "border-border/50 hover:bg-muted text-muted-foreground hover:text-foreground"
                          )}
                        >
                          {t}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Labels */}
                  {allLabels.length > 0 && (
                    <div className="space-y-2.5">
                      <p className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground/70">
                        Labels
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {allLabels.map((label) => (
                          <button
                            key={label}
                            onClick={() => toggleFilter(selectedLabels, setSelectedLabels, label)}
                            className={cn(
                              "rounded-md border px-2.5 py-1.5 text-[11px] font-medium transition-colors outline-none",
                              selectedLabels.has(label)
                                ? "border-primary bg-primary/10 text-primary"
                                : "border-border/50 hover:bg-muted text-muted-foreground hover:text-foreground"
                            )}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Clear */}
                  {activeFilterCount > 0 && (
                    <>
                      <Separator className="my-2" />
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setSelectedPriorities(new Set());
                          setSelectedTypes(new Set());
                          setSelectedLabels(new Set());
                        }}
                        className="w-full text-xs font-semibold text-muted-foreground hover:text-foreground h-8"
                      >
                        Clear active filters
                      </Button>
                    </>
                  )}
                </div>
              </PopoverContent>
            </Popover>

            {/* Clear all indicator */}
            {hasAnyFilter && (
              <>
                <Separator orientation="vertical" className="h-5 hidden sm:block mx-1" />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearAllFilters}
                  className="h-9 px-2.5 text-xs font-semibold text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                >
                  <X className="mr-1.5 size-3.5" />
                  Clear all
                </Button>
                <span className="hidden lg:inline-flex text-xs font-medium text-muted-foreground/60 ml-2">
                  Showing {filteredTickets.length} of {tickets.length}
                </span>
              </>
            )}
          </div>
        ) : (
          <div className="flex flex-1 items-center gap-2">
            <Badge variant="secondary" className="px-2.5 py-1 text-xs font-medium">
              {project.members.length} {project.members.length === 1 ? "Member" : "Members"}
            </Badge>
          </div>
        )}

        {/* Right Side: Settings */}
        <div className="flex items-center shrink-0 ml-4">
          {canManage && (
            <Link href={`/projects/${slug}/settings`}>
              <Button variant="outline" size="sm" className="h-9 bg-background text-[13px] font-medium shadow-sm transition-colors hover:bg-muted">
                <Settings className="mr-2 size-3.5 text-muted-foreground" />
                {activeTab === "team" ? "Manage Team" : "Project Settings"}
              </Button>
            </Link>
          )}
        </div>
      </div>

      {/* ========== FULL BLEED CANVAS AREA ========== */}
      <div className="flex-1 w-full min-w-0 overflow-hidden bg-muted/10 relative">
        {activeTab === "board" && (
          <div className="absolute inset-0 w-full h-full">
            <BoardView
              projectSlug={slug}
              boardColumns={project.board_columns}
              members={project.members}
              sizePointsMap={project.size_points_map}
              weeklyPointsPerMember={project.weekly_points_per_member}
              ticketFilter={(ticket) => {
                if (searchQuery) {
                  const q = searchQuery.toLowerCase();
                  const match =
                    ticket.title.toLowerCase().includes(q) ||
                    ticket.ticket_key.toLowerCase().includes(q) ||
                    (ticket.description || "").toLowerCase().includes(q);
                  if (!match) return false;
                }
                if (selectedMemberIds.size > 0) {
                  if (!ticket.assignee || !selectedMemberIds.has(ticket.assignee.id)) return false;
                }
                if (selectedPriorities.size > 0) {
                  if (!selectedPriorities.has(ticket.priority)) return false;
                }
                if (selectedTypes.size > 0) {
                  if (!selectedTypes.has(ticket.type)) return false;
                }
                if (selectedLabels.size > 0) {
                  const ticketLabels = new Set(ticket.labels || []);
                  if (![...selectedLabels].some((l) => ticketLabels.has(l))) return false;
                }
                return true;
              }}
            />
          </div>
        )}
        {activeTab === "team" && (
          <div className="absolute inset-0 w-full h-full overflow-auto">
            <TeamView
              members={project.members}
              tickets={tickets}
              projectSlug={project.slug}
              sizePointsMap={project.size_points_map}
              weeklyPointsPerMember={project.weekly_points_per_member}
            />
          </div>
        )}
      </div>
    </div>
  );
}