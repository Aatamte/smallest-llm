import { useLeaderboard } from "../db/hooks";
import { LeaderboardPage } from "../components/LeaderboardPage";

export function LeaderboardContainer() {
  const entries = useLeaderboard();
  return <LeaderboardPage entries={entries} />;
}
