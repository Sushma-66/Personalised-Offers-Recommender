-- Do customers in financial difficulty engage MORE with risky products?
-- And what happens to them after they take credit?

SELECT
    cat.item_type,
    cat.risk_level,
    f.financial_difficulty_flag,
    COUNT(*)                                        AS times_shown,
    ROUND(AVG(e.engagement_score), 2)               AS avg_engagement_score,
    ROUND(AVG(CASE WHEN e.engagement_score >= 4 THEN 1.0 ELSE 0 END), 3) AS activation_rate,
    ROUND(AVG(e.missed_payment_within_6m), 3)       AS missed_payment_rate_after_credit
FROM engagement      AS e
JOIN catalogue       AS cat ON cat.item_id = e.item_id
JOIN customer_flags  AS f   ON f.customer_id = e.customer_id
GROUP BY cat.item_type, cat.risk_level, f.financial_difficulty_flag
ORDER BY cat.item_type, cat.risk_level, f.financial_difficulty_flag;
