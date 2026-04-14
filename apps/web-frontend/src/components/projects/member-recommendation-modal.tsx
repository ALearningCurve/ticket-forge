"use client";

import { useEffect, useState } from "react";
import { Briefcase, Loader2, UserRound } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/lib/auth-context";
import {
  getEngineerTicketRecommendations,
  type EngineerTicketRecommendationsResponse,
  type ProjectMember,
} from "@/lib/api";

interface MemberRecommendationModalProps {
  member: ProjectMember | null;
  projectSlug: string;
  open: boolean;
  onClose: () => void;
}

export function MemberRecommendationModal({
  member,
  projectSlug,
  open,
  onClose,
}: MemberRecommendationModalProps) {
  const { token } = useAuth();
  const [data, setData] = useState<EngineerTicketRecommendationsResponse | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    async function loadRecommendations() {
      if (!open || !member || !token) return;

      setIsLoading(true);
      setError(null);
      const result = await getEngineerTicketRecommendations(
        token,
        projectSlug,
        member.user_id
      );
      if (result.error) {
        setData(null);
        setError(result.error);
      } else {
        setData(result.data);
      }
      setIsLoading(false);
    }

    void loadRecommendations();
  }, [member, open, projectSlug, token]);

  return (
    <Dialog open={open} onOpenChange={(value) => !value && onClose()}>
      <DialogContent className="max-h-[85vh] max-w-3xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <UserRound className="size-4" />
            {member
              ? `${member.first_name} ${member.last_name}`
              : "Engineer recommendations"}
          </DialogTitle>
        </DialogHeader>

        {isLoading ? (
          <div className="flex items-center gap-2 py-8 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            Loading recommended tickets...
          </div>
        ) : error ? (
          <p className="py-6 text-sm text-muted-foreground">{error}</p>
        ) : !data ? (
          <p className="py-6 text-sm text-muted-foreground">
            No ticket recommendations available yet.
          </p>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">@{data.username}</Badge>
              <Badge variant={data.has_capacity ? "default" : "secondary"}>
                {data.has_capacity ? "Has capacity" : "At capacity"}
              </Badge>
              <Badge variant="secondary">
                Active tickets: {data.active_ticket_count}
              </Badge>
            </div>

            <Separator />

            <div className="space-y-3">
              {data.recommendations.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No ticket recommendations available yet.
                </p>
              ) : (
                data.recommendations.map((ticket) => (
                  <div
                    key={ticket.ticket_key}
                    className="rounded-lg border bg-muted/20 p-4"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold">{ticket.title}</p>
                        <p className="text-xs text-muted-foreground">
                          {ticket.ticket_key} • {ticket.column_name}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="capitalize">
                          {ticket.priority}
                        </Badge>
                        <Badge variant="outline" className="capitalize">
                          {ticket.type}
                        </Badge>
                      </div>
                    </div>

                    {ticket.description ? (
                      <p className="mt-2 text-sm text-muted-foreground">
                        {ticket.description}
                      </p>
                    ) : null}

                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      {(ticket.labels ?? []).map((label) => (
                        <Badge key={label} variant="outline">
                          {label}
                        </Badge>
                      ))}
                    </div>

                    <div className="mt-3 grid gap-2 text-xs text-muted-foreground md:grid-cols-2">
                      <p>
                        Recommendation:{" "}
                        {Math.round((ticket.recommendation_score ?? 0) * 100)}
                      </p>
                      <p>
                        Semantic: {Math.round((ticket.semantic_similarity ?? 0) * 100)}
                      </p>
                      <p>
                        Lexical: {Math.round((ticket.lexical_score ?? 0) * 100)}
                      </p>
                      <p>
                        {ticket.assignee_name
                          ? `Assigned to ${ticket.assignee_name}`
                          : "Currently unassigned"}
                      </p>
                    </div>

                    {ticket.due_date ? (
                      <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                        <Briefcase className="size-3.5" />
                        Due {new Date(ticket.due_date).toLocaleDateString("en-US")}
                      </div>
                    ) : null}
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
