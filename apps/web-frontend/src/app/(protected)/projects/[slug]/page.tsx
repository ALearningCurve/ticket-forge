"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  Filter,
  Loader2,
  Search,
  Settings,
} from "lucide-react";
import { toast } from "sonner";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { BoardView } from "@/components/projects/board/board-view";
import { TeamView } from "@/components/projects/team-view";
import { useAuth } from "@/lib/auth-context";
import {
  getProject,
  getBoardTickets,
  type ProjectResponse,
  type TicketResponse,
} from "@/lib/api";

type TabType = "board" | "team";

export default function ProjectDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { token, user } = useAuth();
  const slug = params.slug as string;

  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [tickets, setTickets] = useState<TicketResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabType>("board");

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

  // Re-fetch tickets when switching to team tab (fresh data)
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

  const myRole = project?.members.find(
    (m) => m.user_id === user?.id
  )?.role;
  const canManage = myRole === "owner" || myRole === "admin";

  if (isLoading) {
    return (
      <div className="flex h-[calc(100vh-4rem)] items-center justify-center">
        <Loader2 className="size-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!project) return null;

  const memberColors = [
    "#6366f1",
    "#06b6d4",
    "#10b981",
    "#f59e0b",
    "#ef4444",
  ];

  const tabs: { key: TabType; label: string }[] = [
    { key: "board", label: "Board" },
    { key: "team", label: "Team" },
  ];

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col overflow-hidden">
      {/* Breadcrumb + Tabs */}
      <div className="border-b px-6">
        <div className="flex items-center gap-1.5 pt-2.5 text-[13px]">
          <Link
            href="/dashboard"
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            Projects
          </Link>
          <span className="text-muted-foreground/50">/</span>
          <span className="font-medium">{project.name}</span>
        </div>

        {/* Tabs */}
        <div className="mt-2 flex gap-0">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => handleTabChange(tab.key)}
              className={`relative px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === tab.key
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {tab.label}
              {activeTab === tab.key && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Toolbar (board only) */}
      {activeTab === "board" && (
        <div className="flex items-center justify-between border-b px-6 py-2">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search board"
                className="h-8 w-48 pl-8 text-[13px]"
              />
            </div>

            <div className="flex -space-x-1.5">
              {project.members.slice(0, 4).map((member, idx) => (
                <button
                  key={member.id}
                  className="flex size-7 items-center justify-center rounded-full border-2 border-background text-[10px] font-semibold text-white transition-transform hover:z-10 hover:scale-110"
                  style={{
                    backgroundColor: memberColors[idx % memberColors.length],
                  }}
                  title={`${member.first_name} ${member.last_name}`}
                >
                  {member.first_name[0]}
                  {member.last_name[0]}
                </button>
              ))}
              {project.members.length > 4 && (
                <div className="flex size-7 items-center justify-center rounded-full border-2 border-background bg-muted text-[10px] font-medium text-muted-foreground">
                  +{project.members.length - 4}
                </div>
              )}
            </div>

            <Separator orientation="vertical" className="h-5" />

            <Button variant="ghost" size="sm" className="h-8 text-[13px]">
              <Filter className="mr-1.5 size-3.5" />
              Filter
            </Button>
          </div>

          <div className="flex items-center gap-2">
            {canManage && (
              <Link href={`/projects/${slug}/settings`}>
                <Button variant="ghost" size="sm" className="h-8 text-[13px]">
                  <Settings className="mr-1.5 size-3.5" />
                  Settings
                </Button>
              </Link>
            )}
          </div>
        </div>
      )}

      {/* Team toolbar */}
      {activeTab === "team" && (
        <div className="flex items-center justify-between border-b px-6 py-2">
          <p className="text-sm text-muted-foreground">
            {project.members.length}{" "}
            {project.members.length === 1 ? "member" : "members"}
          </p>
          {canManage && (
            <Link href={`/projects/${slug}/settings`}>
              <Button variant="ghost" size="sm" className="h-8 text-[13px]">
                <Settings className="mr-1.5 size-3.5" />
                Manage team
              </Button>
            </Link>
          )}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-x-auto overflow-y-auto bg-background">
        {activeTab === "board" && (
          <div className="h-full p-5">
            <BoardView
              projectSlug={slug}
              boardColumns={project.board_columns}
              members={project.members}
              sizePointsMap={project.size_points_map}
              weeklyPointsPerMember={project.weekly_points_per_member}
            />
          </div>
        )}
        {activeTab === "team" && (
          <TeamView
            members={project.members}
            tickets={tickets}
            projectSlug={project.slug}
            sizePointsMap={project.size_points_map}
            weeklyPointsPerMember={project.weekly_points_per_member}
          />
        )}
      </div>
    </div>
  );
}