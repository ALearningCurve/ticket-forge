"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ArrowLeft,
  Columns3,
  GripVertical,
  Loader2,
  Plus,
  Ruler,
  Save,
  Target,
  Trash2,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";

import { useAuth } from "@/lib/auth-context";
import { MemberSearch } from "@/components/projects/member-search";
import {
  addProjectMember,
  deleteProject,
  getProject,
  removeProjectMember,
  updateMemberRole,
  type ProjectResponse,
  type UserSearchResult,
} from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const sizeOptions = [
  { value: "S", label: "Small" },
  { value: "M", label: "Medium" },
  { value: "L", label: "Large" },
  { value: "XL", label: "Extra Large" },
];

export default function ProjectSettingsPage() {
  const params = useParams();
  const router = useRouter();
  const { token, user } = useAuth();
  const slug = params.slug as string;

  const [project, setProject] = useState<ProjectResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // General
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [defaultTicketSize, setDefaultTicketSize] = useState("M");
  const [isSaving, setIsSaving] = useState(false);

  // Capacity
  const [weeklyPoints, setWeeklyPoints] = useState(10);
  const [sizePoints, setSizePoints] = useState({ S: 1, M: 2, L: 3, XL: 5 });
  const [isSavingCapacity, setIsSavingCapacity] = useState(false);

  // Columns
  const [columns, setColumns] = useState<{ id?: string; name: string }[]>([]);
  const [isSavingColumns, setIsSavingColumns] = useState(false);

  useEffect(() => {
    async function load() {
      if (!token) return;
      const { data, error } = await getProject(token, slug);
      if (error) {
        toast.error(error);
        router.push("/dashboard");
        return;
      }
      if (data) {
        setProject(data);
        setName(data.name);
        setDescription(data.description || "");
        setDefaultTicketSize(data.default_ticket_size || "M");
        setWeeklyPoints(data.weekly_points_per_member || 10);
        setSizePoints(data.size_points_map || { S: 1, M: 2, L: 3, XL: 5 });
        setColumns(
          data.board_columns
            .sort((a, b) => a.position - b.position)
            .map((c) => ({ id: c.id, name: c.name }))
        );
      }
      setIsLoading(false);
    }
    load();
  }, [token, slug, router]);

  const myRole = project?.members.find(
    (m) => m.user_id === user?.id
  )?.role;
  const canManage = myRole === "owner" || myRole === "admin";
  const isOwner = myRole === "owner";

  async function handleSaveDetails() {
    if (!token || !project) return;
    setIsSaving(true);

    const res = await fetch(`${API_BASE}/api/v1/projects/${slug}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      credentials: "include",
      body: JSON.stringify({
        name: name.trim() || undefined,
        description: description.trim(),
        default_ticket_size: defaultTicketSize,
      }),
    });

    setIsSaving(false);
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      toast.error(err?.detail || "Failed to update project");
      return;
    }

    const updated = await res.json();
    setProject(updated);
    toast.success("Project updated");

    if (updated.slug !== slug) {
      router.replace(`/projects/${updated.slug}/settings`);
    }
  }

  async function handleSaveCapacity() {
    if (!token || !project) return;
    setIsSavingCapacity(true);

    const res = await fetch(`${API_BASE}/api/v1/projects/${slug}`, {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      credentials: "include",
      body: JSON.stringify({
        weekly_points_per_member: weeklyPoints,
        size_points_map: sizePoints,
      }),
    });

    setIsSavingCapacity(false);
    if (!res.ok) {
      const err = await res.json().catch(() => null);
      toast.error(err?.detail || "Failed to update capacity settings");
      return;
    }

    const updated = await res.json();
    setProject(updated);
    toast.success("Capacity settings saved");
  }

  function addColumn() {
    if (columns.length >= 12) return;
    setColumns([...columns, { name: "" }]);
  }

  function removeColumn(index: number) {
    if (columns.length <= 1) return;
    setColumns(columns.filter((_, i) => i !== index));
  }

  function updateColumnName(index: number, value: string) {
    const updated = [...columns];
    updated[index] = { ...updated[index], name: value };
    setColumns(updated);
  }

  async function handleSaveColumns() {
    const nonEmpty = columns.filter((c) => c.name.trim());
    if (nonEmpty.length === 0) {
      toast.error("Add at least one board column");
      return;
    }
    const names = nonEmpty.map((c) => c.name.trim().toLowerCase());
    if (new Set(names).size !== names.length) {
      toast.error("Column names must be unique");
      return;
    }
    setIsSavingColumns(true);
    await new Promise((r) => setTimeout(r, 500));
    setIsSavingColumns(false);
    toast.success("Board columns saved");
  }

  async function handleAddMember(memberUser: UserSearchResult) {
    if (!token || !project) return;
    const { data, error } = await addProjectMember(token, project.slug, {
      user_id: memberUser.id,
      role: "member",
    });
    if (error) {
      toast.error(error);
      return;
    }
    if (data) {
      setProject((prev) =>
        prev ? { ...prev, members: [...prev.members, data] } : null
      );
      toast.success(`Added ${memberUser.first_name}`);
    }
  }

  async function handleRemoveMember(userId: string) {
    if (!token || !project) return;
    const member = project.members.find((m) => m.user_id === userId);
    const confirmed = window.confirm(
      `Remove ${member?.first_name} ${member?.last_name} from this project?`
    );
    if (!confirmed) return;

    const { error } = await removeProjectMember(token, project.slug, userId);
    if (error) {
      toast.error(error);
      return;
    }
    setProject((prev) =>
      prev
        ? { ...prev, members: prev.members.filter((m) => m.user_id !== userId) }
        : null
    );
    toast.success("Member removed");
  }

  async function handleChangeRole(userId: string, newRole: string) {
    if (!token || !project) return;
    const { data, error } = await updateMemberRole(
      token,
      project.slug,
      userId,
      newRole
    );
    if (error) {
      toast.error(error);
      return;
    }
    if (data) {
      setProject((prev) =>
        prev
          ? {
              ...prev,
              members: prev.members.map((m) =>
                m.user_id === userId ? { ...m, role: data.role } : m
              ),
            }
          : null
      );
      toast.success("Role updated");
    }
  }

  async function handleDelete() {
    if (!token || !project) return;
    const confirmed = window.confirm(
      `Delete "${project.name}"? This cannot be undone.`
    );
    if (!confirmed) return;

    const { error } = await deleteProject(token, project.slug);
    if (error) {
      toast.error(error);
      return;
    }
    toast.success("Project deleted");
    router.push("/dashboard");
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!project) return null;

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <Link
        href={`/projects/${slug}`}
        className="mb-4 inline-flex items-center text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="mr-1 size-3.5" />
        Back to {project.name}
      </Link>

      <h1 className="text-2xl font-bold tracking-tight">Project settings</h1>
      <p className="text-sm text-muted-foreground">
        Manage your project details, capacity, board, and team.
      </p>

      <Separator className="my-6" />

      {/* ---- General ---- */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">General</CardTitle>
          <CardDescription>
            Update project name, description, and default ticket size.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="edit-name">Project name</Label>
            <Input
              id="edit-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={!canManage}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="edit-desc">Description</Label>
            <textarea
              id="edit-desc"
              className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              disabled={!canManage}
              maxLength={500}
            />
          </div>
          <div className="space-y-2">
            <Label>Default ticket size</Label>
            <Select
              value={defaultTicketSize}
              onValueChange={setDefaultTicketSize}
              disabled={!canManage}
            >
              <SelectTrigger className="w-full max-w-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {sizeOptions.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    <span className="font-semibold">{opt.value}</span>
                    <span className="ml-2 text-muted-foreground">
                      {opt.label}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
        {canManage && (
          <CardFooter className="justify-end">
            <Button onClick={handleSaveDetails} disabled={isSaving} size="sm">
              {isSaving ? (
                <Loader2 className="mr-1.5 size-3.5 animate-spin" />
              ) : (
                <Save className="mr-1.5 size-3.5" />
              )}
              Save changes
            </Button>
          </CardFooter>
        )}
      </Card>

      <Separator className="my-6" />

      {/* ---- Sprint Capacity ---- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Target className="size-4" />
            Sprint capacity
          </CardTitle>
          <CardDescription>
            Set the weekly point budget per member and how many points each
            ticket size is worth. The AI recommendation system uses this to
            balance workload across the team.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Weekly points */}
          <div className="space-y-2">
            <Label htmlFor="weekly-points">
              Weekly points per member
            </Label>
            <p className="text-xs text-muted-foreground">
              Each team member can be assigned up to this many points of work
              per week.
            </p>
            <div className="flex items-center gap-3">
              <Input
                id="weekly-points"
                type="number"
                min={1}
                max={100}
                value={weeklyPoints}
                onChange={(e) =>
                  setWeeklyPoints(parseInt(e.target.value, 10) || 1)
                }
                disabled={!canManage}
                className="w-24"
              />
              <span className="text-sm text-muted-foreground">
                points / week
              </span>
            </div>
          </div>

          <Separator />

          {/* Size point values */}
          <div className="space-y-3">
            <Label>Points per ticket size</Label>
            <p className="text-xs text-muted-foreground">
              Define how many points each ticket size costs. Larger tickets
              consume more of a member&apos;s weekly budget.
            </p>

            <div className="grid grid-cols-4 gap-3">
              {(["S", "M", "L", "XL"] as const).map((sz) => (
                <div key={sz} className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <Badge
                      variant="secondary"
                      className="text-xs font-bold"
                    >
                      {sz}
                    </Badge>
                  </div>
                  <Input
                    type="number"
                    min={0}
                    max={100}
                    value={sizePoints[sz]}
                    onChange={(e) =>
                      setSizePoints({
                        ...sizePoints,
                        [sz]: parseInt(e.target.value, 10) || 0,
                      })
                    }
                    disabled={!canManage}
                    className="text-center"
                  />
                  <p className="text-center text-[10px] text-muted-foreground">
                    {sizePoints[sz] === 1
                      ? "1 point"
                      : `${sizePoints[sz]} points`}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Visual summary */}
          <div className="rounded-lg border bg-muted/30 p-4">
            <p className="mb-2 text-xs font-medium text-muted-foreground">
              Capacity example
            </p>
            <p className="text-sm">
              With <span className="font-semibold">{weeklyPoints} pts/week</span>,
              a member can handle:
            </p>
            <div className="mt-2 flex flex-wrap gap-2">
              {(["S", "M", "L", "XL"] as const).map((sz) => {
                const pts = sizePoints[sz];
                const count = pts > 0 ? Math.floor(weeklyPoints / pts) : 0;
                return (
                  <Badge key={sz} variant="outline" className="text-xs">
                    {count}× {sz}
                    <span className="ml-1 text-muted-foreground">
                      ({pts}pt each)
                    </span>
                  </Badge>
                );
              })}
            </div>
          </div>
        </CardContent>
        {canManage && (
          <CardFooter className="justify-end">
            <Button
              onClick={handleSaveCapacity}
              disabled={isSavingCapacity}
              size="sm"
            >
              {isSavingCapacity ? (
                <Loader2 className="mr-1.5 size-3.5 animate-spin" />
              ) : (
                <Save className="mr-1.5 size-3.5" />
              )}
              Save capacity
            </Button>
          </CardFooter>
        )}
      </Card>

      <Separator className="my-6" />

      {/* ---- Board Columns ---- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Columns3 className="size-4" />
            Board columns
          </CardTitle>
          <CardDescription>
            Configure your kanban board columns and their order.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {columns.map((col, index) => (
              <div key={index} className="flex items-center gap-2">
                <GripVertical className="size-4 shrink-0 text-muted-foreground" />
                <Input
                  value={col.name}
                  onChange={(e) => updateColumnName(index, e.target.value)}
                  placeholder={`Column ${index + 1}`}
                  disabled={!canManage}
                  className="flex-1"
                />
                {canManage && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => removeColumn(index)}
                    disabled={columns.length <= 1}
                    className="shrink-0"
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                )}
              </div>
            ))}
          </div>
          {canManage && columns.length < 12 && (
            <Button
              variant="outline"
              size="sm"
              onClick={addColumn}
              className="mt-3 w-full"
            >
              <Plus className="mr-1.5 size-3.5" />
              Add column
            </Button>
          )}
        </CardContent>
        {canManage && (
          <CardFooter className="justify-end">
            <Button
              onClick={handleSaveColumns}
              disabled={isSavingColumns}
              size="sm"
            >
              {isSavingColumns ? (
                <Loader2 className="mr-1.5 size-3.5 animate-spin" />
              ) : (
                <Save className="mr-1.5 size-3.5" />
              )}
              Save columns
            </Button>
          </CardFooter>
        )}
      </Card>

      <Separator className="my-6" />

      {/* ---- Members ---- */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Users className="size-4" />
            Members ({project.members.length})
          </CardTitle>
          <CardDescription>
            Manage who has access to this project.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {canManage && (
            <MemberSearch
              selected={[]}
              onSelect={handleAddMember}
              onRemove={() => {}}
              projectSlug={project.slug}
            />
          )}

          <div className="space-y-2">
            {project.members.map((member) => (
              <div
                key={member.id}
                className="flex items-center justify-between rounded-md border px-3 py-2.5"
              >
                <div className="flex items-center gap-3">
                  <div className="flex size-8 items-center justify-center rounded-full bg-muted text-xs font-medium">
                    {member.first_name[0]}
                    {member.last_name[0]}
                  </div>
                  <div>
                    <p className="text-sm font-medium">
                      {member.first_name} {member.last_name}
                      {member.user_id === user?.id && (
                        <span className="ml-1.5 text-xs text-muted-foreground">
                          (you)
                        </span>
                      )}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {member.email}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {isOwner && member.role !== "owner" ? (
                    <select
                      value={member.role}
                      onChange={(e) =>
                        handleChangeRole(member.user_id, e.target.value)
                      }
                      className="h-7 rounded-md border bg-background px-2 text-xs capitalize"
                    >
                      <option value="admin">Admin</option>
                      <option value="member">Member</option>
                    </select>
                  ) : (
                    <Badge variant="secondary" className="capitalize">
                      {member.role}
                    </Badge>
                  )}

                  {canManage &&
                    member.role !== "owner" &&
                    member.user_id !== user?.id && (
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        onClick={() => handleRemoveMember(member.user_id)}
                      >
                        <Trash2 className="size-3" />
                      </Button>
                    )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* ---- Danger Zone ---- */}
      {isOwner && (
        <>
          <Separator className="my-6" />
          <Card className="border-destructive/50">
            <CardHeader>
              <CardTitle className="text-base text-destructive">
                Danger zone
              </CardTitle>
              <CardDescription>
                Permanently delete this project and all its data.
              </CardDescription>
            </CardHeader>
            <CardFooter>
              <Button variant="destructive" size="sm" onClick={handleDelete}>
                <Trash2 className="mr-1.5 size-3.5" />
                Delete project
              </Button>
            </CardFooter>
          </Card>
        </>
      )}
    </div>
  );
}