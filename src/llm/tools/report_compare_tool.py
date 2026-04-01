"""ReportCompareTool — compare two intelligence reports and identify changes."""

from typing import Any, Dict

from .base import BaseTool, ToolResult
from ...services.report_compare_service import compare_reports
from ...utils.logger import get_logger

logger = get_logger(__name__)


class ReportCompareTool(BaseTool):
    """Compare two intelligence reports and generate delta analysis."""

    name = "report_compare"
    description = (
        "Compare two intelligence reports and identify what changed between them. "
        "Generates structured analysis of new developments, resolved topics, trend shifts, and persistent themes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "rationale": {
                "type": "string",
                "description": "PATH COMPARATIVE: Explain which two reports you're comparing and what delta you expect to find (new developments, trend shifts, resolved topics).",
            },
            "report_id_a": {
                "type": "integer",
                "description": "ID of the older/first report to compare",
            },
            "report_id_b": {
                "type": "integer",
                "description": "ID of the newer/second report to compare",
            },
        },
        "required": ["rationale", "report_id_a", "report_id_b"],
    }

    def _execute(self, **kwargs) -> ToolResult:
        """Execute report comparison."""
        report_id_a: int = kwargs.get("report_id_a")
        report_id_b: int = kwargs.get("report_id_b")

        if not report_id_a or not report_id_b:
            return ToolResult(
                success=False,
                data=None,
                error="Both report_id_a and report_id_b are required",
            )

        try:
            # Fetch reports from database
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Fetch report A
                    cur.execute("""
                        SELECT id, report_date, report_type, status,
                               draft_content, final_content
                        FROM reports
                        WHERE id = %s
                    """, [report_id_a])
                    row_a = cur.fetchone()

                    # Fetch report B
                    cur.execute("""
                        SELECT id, report_date, report_type, status,
                               draft_content, final_content
                        FROM reports
                        WHERE id = %s
                    """, [report_id_b])
                    row_b = cur.fetchone()

            if not row_a or not row_b:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Report not found (A={report_id_a}, B={report_id_b})",
                )

            # Build report dicts
            report_a = {
                'id': row_a[0],
                'report_date': row_a[1],
                'report_type': row_a[2],
                'status': row_a[3],
                'draft_content': row_a[4],
                'final_content': row_a[5],
            }
            report_b = {
                'id': row_b[0],
                'report_date': row_b[1],
                'report_type': row_b[2],
                'status': row_b[3],
                'draft_content': row_b[4],
                'final_content': row_b[5],
            }

            # Call comparison service
            delta = compare_reports(report_a, report_b)

            logger.info(
                f"ReportCompareTool: Compared reports {report_id_a} and {report_id_b} successfully"
            )

            return ToolResult(
                success=True,
                data=delta,
                metadata={
                    "report_a_id": report_id_a,
                    "report_b_id": report_id_b,
                    "sections": len([x for v in delta.values() for x in v]),
                },
            )

        except Exception as e:
            logger.error(f"ReportCompareTool: Execution error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to compare reports: {str(e)}",
            )

    def _format_success(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Format comparison delta result for LLM consumption."""
        if not data:
            return "No delta analysis available."

        lines = []

        # New developments
        new_dev = data.get("new_developments", [])
        if new_dev:
            lines.append("**NEW DEVELOPMENTS:**")
            for item in new_dev:
                lines.append(f"  • {item}")
            lines.append("")

        # Resolved topics
        resolved = data.get("resolved_topics", [])
        if resolved:
            lines.append("**RESOLVED TOPICS:**")
            for item in resolved:
                lines.append(f"  • {item}")
            lines.append("")

        # Trend shifts
        shifts = data.get("trend_shifts", [])
        if shifts:
            lines.append("**TREND SHIFTS:**")
            for item in shifts:
                lines.append(f"  • {item}")
            lines.append("")

        # Persistent themes
        persistent = data.get("persistent_themes", [])
        if persistent:
            lines.append("**PERSISTENT THEMES:**")
            for item in persistent:
                lines.append(f"  • {item}")

        return "\n".join(lines) if lines else "No significant changes identified."
