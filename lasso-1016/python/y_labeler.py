#!/usr/bin/env python3
"""冻结的中文弱标签规则 v1.0.3；规则输出是训练标签，不是人工金标准。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

VERSION = "zh_consumer_semantics_20261012_new_v1.0.3"
HERE = Path(__file__).resolve().parent
COMPONENTS = ("c06", "c07", "c12", "c16", "c25", "c28", "c34", "c41", "c42", "c43")
OUTCOMES = {
    "y01": ["c41"], "y02": ["c43"], "y03": ["c28"],
    "y04": ["c12", "c16"], "y05": ["c06", "c07"],
    "y06": ["c16"], "y07": ["c42"], "y08": ["c12"],
    "y09": ["c25"], "y10": ["c34"],
}
LEGACY_OUTCOME_IDS = {"y04": "r2_07", "y05": "r2_03"}

# These are explicitly narrow lexical anchors, not learned features.
PROMO = r"双十一|双11|双十二|双12|618|促销|大促|优惠|折扣|满减|凑单|拼单|红包|券|秒杀|补贴|淘金币|返利|购物攻略|购物规则|促销规则|活动规则|积分(?:兑换|换|抵扣)|组队.{0,6}(?:积分|红包)"
SHOP = PROMO + r"|购物|买(?:东西|什么|哪|手机|衣服|鞋|了)?|购买|下单|付款|消费|商品|产品|商家|店铺|卖家|价格|价钱|运费|抵扣|津贴|兑换|淘宝|京东|拼多多|天猫|得物"
PRODUCT = r"商品|产品|东西|这款|这个牌子|手机|电脑|耳机|面霜|护肤|拖鞋|鞋子|衣服|洗发水|口红|化妆品|电器|食品|饮料|香水|牙膏|洗衣液|面膜|牛奶|屏幕|电池"
DURATION = r"(?:很久|好久|半天|老半天|一天|一整天|几个小时|好几个小时|很多时间|不少时间|大量时间|时间|[一二三四五六七八九十两\d.]+(?:个)?(?:小时|分钟|天|夜))"
NUMERIC_DURATION = r"(?:半(?:个)?月|半天|[一二三四五六七八九十两\d]+(?:个)?(?:小时|分钟|天|周|星期|月))"
CLAUSE_BREAK = re.compile(r"[，,。.!！?？;；\n]")

RULES = {
    "c06": [
        ("time_spent", rf"(?:花|耗|费|用了?|浪费|折腾)(?:了|掉|费了)?(?:整整|足足)?{DURATION}"),
        ("study_duration", rf"(?:研究|琢磨|计算|算|看|凑单|凑)(?:了|来凑去|半)?(?:好久|很久|半天|老半天|[一二三四五六七八九十两\d]+(?:个)?小时)"),
        ("measured_study_duration", rf"(?:研究|琢磨|计算|凑单|盯|观察|看)(?:了)?{NUMERIC_DURATION}"),
        ("time_labor", r"耗时|(?<!浪)费时|熬夜|定(?:好|了)?闹钟|设(?:置|了)?闹钟|守到(?:凌晨|半夜)|蹲到(?:凌晨|半夜)|凌晨.{0,6}抢"),
    ],
    "c07": [
        ("scarcity", r"没(?:有)?时间|没(?:有)?空|无暇|抽不出时间|忙得没空|忙到没空|哪有时间|哪来的时间"),
        ("opportunity_cost", r"时间(?:更|也|很|太)?值钱|时间成本(?:太|很)?高|不值得(?:浪费|花)(?:这个|这)?时间|浪费时间|没工夫|没功夫"),
    ],
    "c12": [
        ("price_rise", r"涨价|提价|涨成\s*\d+(?:\.\d+)?|涨到\s*\d+(?:\.\d+)?|涨了\s*(?:[一二三四五六七八九十百千万两\d.]+)(?:块|元|倍|%)?"),
        ("fake_discount", r"先(?:涨|提)(?:价)?(?:再|后)(?:降|打折|减)|先涨后降|先提后降|(?:原价|价格).{0,8}(?:抬高|上调)|(?:抬高|上调).{0,8}(?:原价|价格)"),
    ],
    "c16": [
        ("ironic_effort_reward", r"一顿操作猛如虎[，,\s]*(?:一看|结果)(?:才|只|就)?(?:省了|省下|便宜了)[\d一二三四五六七八九十两几.]+(?:块|元)(?:[\d一二三四五六七八九十两几.]+)?"),
        ("saving_failure", r"没(?:有)?省(?:下|到)?(?:钱|多少|几块|[\d一二三四五六七八九十两]+块)|省不(?:了|下|到)(?:钱|多少|几块|[\d一二三四五六七八九十两]+块)?|一点(?:钱)?(?:也|都)?没省|没(?:有)?便宜(?:多少|几块|[\d一二三四五六七八九十两]+块)?"),
        ("bad_value", r"不(?:太)?划算|不值(?:得|当|这个价|这价)|不合算|不划得来|不便宜|越买越贵|比[^，,。.!！?？;；\n]{1,16}(?:还|更)贵|反而(?:更|还)?贵"),
        ("extra_spending", r"(?:反而|结果|最后|又|还|白白|多)(?:多)?花(?:了)?(?:冤枉钱|钱|[\d一二三四五六七八九十百千万两]+(?:块|元))|花(?:了)?冤枉钱|多花"),
    ],
    "c25": [
        ("completed_buy", r"买了|买过|买完了?|买下了?|购入了|购入过|购买了|购买过|入手了|入手过|拍下了|拍了|下单了|下过单|已(?:经)?(?:购买|买|购|下单|入手)|刚(?:刚)?下单"),
    ],
    "c28": [
        ("abstention_saves", r"不买[^，,。.!！?？;；\n]{0,14}(?:最省钱|最便宜|省钱|立省|省下|省了|省100%|省百分之百)|不花钱[^，,。.!！?？;；\n]{0,8}(?:省钱|最便宜|立省)|不消费[^，,。.!！?？;；\n]{0,8}(?:省钱|立省)|不买就是赚|不买即省"),
    ],
    "c34": [
        ("help_request", r"请问|求助|求推荐|求(?:个|一个)?(?:教程|攻略|方法)|教教(?:我|我们)?|谁(?:能|可以).{0,8}(?:教|解释|说说|告诉)|能不能.{0,6}(?:告诉|教|解释)|能否.{0,6}(?:告诉|教|解释)|告诉我|求解|求解答|不懂怎么|不会(?:用|选|买|操作)|怎么(?:买|选|领|用|凑|算|叠加|兑换|抵扣|操作)|如何(?:买|选|领|用|凑|叠加|兑换|抵扣)|选哪个|买哪个|哪个好|哪款好"),
    ],
    "c41": [
        ("quality_authenticity", r"质量|假货|正品|真货|真假|保质期|变质|过期|日期不新鲜|日期新鲜|以次充好|偷工减料"),
        ("product_effect", r"效果|功效|耐用|好用|难用|没用|无效|有效"),
    ],
    "c42": [
        ("delivery_aftersales", r"物流|快递|配送|发货|收货|到货|售后|退货|退款|保价|价保|价格保护|退换货|换货"),
        ("customer_service", r"客服"),
    ],
    "c43": [
        ("bounded_follow_buy", r"(?:照着|跟着)(?:你|您|他|她|博主|作者)(?:的)?(?:购物清单|清单|攻略|推荐|建议)(?:直接|一起)?(?:买了|买过|购买了|购买过|下单了|入手了)"),
        ("trust_endorsement", r"靠谱|可信|信任|信得过|相信(?:你|您|这个博主|这位博主|这个商家)|放心买"),
        ("recommend_endorsement", r"推荐|值得买|值得入手|闭眼入|闭眼买|跟着(?:你|您|博主|他|她|作者)?买|照着(?:你|您|博主|他|她|作者)?买|抄(?:你|您|博主|他|她|作者)?(?:的)?作业"),
    ],
}


def normalize(text: str):
    """Per-codepoint NFKC, ASCII lowercase, CR->LF, remove BOM/ZWSP/ZWJ/ZWNJ.

    Other whitespace and punctuation are preserved (not tokenized or stripped).
    Per-codepoint operation intentionally does not compose combining marks across
    source codepoints. All offsets and +/-45 windows use normalized codepoints.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str; missing values are not silently stringified")
    chars, source_map = [], []
    for i, ch in enumerate(text):
        if ch in "\ufeff\u200b\u200c\u200d":
            continue
        s = unicodedata.normalize("NFKC", ch)
        if s == "\r":
            s = "\n"
        for c in s:
            chars.append(c.lower() if "A" <= c <= "Z" else c)
            source_map.append(i)
    return "".join(chars), source_map


def _clause(t, start, end):
    left, right = 0, len(t)
    for m in CLAUSE_BREAK.finditer(t):
        if m.end() <= start:
            left = m.end()
        elif m.start() >= end:
            right = m.start()
            break
    return t[left:right], left, right


def _proposition(t, start, end):
    """Clause-local predicate scope; contrast separates independently asserted acts."""
    clause, left, right = _clause(t, start, end)
    scope_left = left
    for m in re.finditer(r"但是|不过|然而|可是|但", clause):
        a, b = left + m.start(), left + m.end()
        if b <= start:
            scope_left = b
        elif a >= end:
            right = a
            break
    # Keep original left offsets while iterating above.
    left = scope_left
    return t[left:right], left, right


def _denied_time(pre):
    return bool(re.search(r"(?:不用|不必|无需|不需要|没有|没|并非|不是|不)(?:再|去|多|怎么|真的|白白|额外|特意|为此|还|也|就|会|要){0,4}$", pre))


def _rejected_report(pre, post):
    """Reject a claim under explicit denial/challenge, not quotations in general."""
    direct = re.search(r"(?:别|不要|别再|不要再)(?:总|老|再|一直)?(?:说|声称|称|觉得|认为)[“\"‘']?$", pre)
    attitude = re.search(r"(?:不认为|不觉得|不相信|不承认|否认)[^，,。！？?!；;]{0,18}[“\"‘']?$", pre)
    challenged_quote = re.search(r"(?:那些|这些|有人|别人|他们).{0,8}(?:说|声称)[“\"‘']?$", pre) and re.match(r"[”\"’']?(?:的)?(?:人)?(?:不知道.{0,6}(?:谁|哪)|没有依据|毫无依据|不属实|说法不实|纯属谣言)", post)
    return bool(direct or attitude or challenged_quote)


def _rejected_quoted_proposition(text, start, end):
    """Only an enclosing quote plus a directly following explicit rejection.

    This is per candidate: a later rejection cannot veto an independent assertion
    outside that quote. Attribution or quotation without rejection is not a veto.
    """
    for quote in re.finditer(r'“[^”\n]{1,120}”|‘[^’\n]{1,120}’|"[^"\n]{1,120}"', text):
        if quote.start() < start and end < quote.end():
            after = text[quote.end():quote.end()+60]
            rejected = re.match(
                r"[\s，,]*(?:但是|不过|然而|可是|但)?"
                r"(?:(?:我|我们|本人|这个说法|这种说法|这句话|上述说法|这点)(?:也|都)?)?"
                r"(?:并|完全|根本|其实|真的|确实)?"
                r"(?:不认同|不认可|不相信|没有认可|没认可|没有认同|没认同|不属实|不是真的|是假的|纯属谣言|毫无依据)",
                after)
            if rejected:
                return {"quote_span": list(quote.span()),
                        "rejection_span": [quote.end(), quote.end()+rejected.end()],
                        "rejection_text": rejected.group()}
    return None


def _denied_self_purchase_report(pre):
    return bool(re.search(
        r"(?:没|没有|并未|不曾|从没|从未)(?:说|表示|声称|承认)(?:过)?"
        r"(?:自己|我|我们|本人|已经|曾经|真的|实际|当时|早就|早已|昨天|刚才|去年|上次){0,6}$", pre))


def _follow_buy_regret(text, start, end):
    """Immediate evaluation of this following-purchase, not whole-text sentiment."""
    tail = text[end:end+64]
    # Existing shorter 跟着…买 candidates leave perfective/object words in tail.
    # Only the directly adjacent comma/contrast assessment may qualify.
    match = re.match(
        r"[^，,。.!！?？;；\n]{0,16}?(?:[，,]\s*|但是|不过|但)"
        r"(?:但是|不过|但|结果|后来)?(?:我|我们|本人|他|她)?"
        r"(?:很|非常|太|真|真的|却|也|十分|特别|现在|有点){0,3}"
        r"(?:后悔|不满意|踩雷|上当)", tail)
    return bool(match)


def _anchors(t, start, end, pattern, radius=45):
    lo, hi = max(0, start-radius), min(len(t), end+radius)
    return [{"text": m.group(), "span": [lo+m.start(), lo+m.end()]} for m in re.finditer(pattern, t[lo:hi])]


def _pre_negated(pre):
    return bool(re.search(r"(?:不|没|没有|未|别|并非|谈不上|不算|说不上)(?:是|太|很|够|怎么|敢|再|会|要|那么|这么|让你|让人){0,4}$", pre))


def _actor(t, start, end):
    clause, left, right = _clause(t, start, end)
    pre = t[max(left, start-24):start]
    actors = list(re.finditer(r"我们|咱们|本人|我|他|她|他们|她们|朋友|妈妈|爸爸|同事|博主|作者|别人|你们|你|您", pre))
    if not actors:
        return "unspecified", None
    token = actors[-1].group()
    if token in ("我", "本人"):
        return "first_person", token
    if token in ("我们", "咱们"):
        return "first_person_collective", token
    if token in ("你", "您", "你们"):
        return "second_person", token
    return "other_person", token


def _qualify(component, rule, text, start, end):
    phrase = text[start:end]
    clause, left, right = _clause(text, start, end)
    pre = text[max(left, start-24):start]
    post = text[end:min(right, end+24)]
    local = text[max(0, start-45):min(len(text), end+45)]
    evidence = {}
    if component in ("c06", "c07", "c12", "c16", "c25", "c34", "c41"):
        clause, left, right = _proposition(text, start, end)
        pre = text[max(left, start-24):start]
        post = text[end:min(right, end+24)]
        evidence["proposition_scope_span"] = [left, right]
    if component in ("c12", "c16", "c43"):
        rejected_quote = _rejected_quoted_proposition(text, start, end)
        if rejected_quote:
            evidence["rejected_quotation"] = rejected_quote
            return False, "excluded:quoted_proposition_explicitly_rejected_by_speaker", evidence
    if component == "c06":
        evidence["anchors"] = _anchors(text, start, end, PROMO)
        if not evidence["anchors"]:
            return False, "excluded:no_promotion_or_rule_study_anchor_within_45", evidence
        if _denied_time(pre):
            return False, "excluded:negated_time_investment", evidence
        if re.search(r"(?:拍|录|制作)(?:这个|这段)?视频|游戏(?:玩|打)|(?:商品|手机|电池|产品).{0,6}(?:使用|续航)", clause) and not re.search(r"(?:研究|计算|凑单|抢券|抢红包)", clause):
            return False, "excluded:content_creation_game_or_product_use_duration", evidence
        if rule == "study_duration" and phrase.startswith("看") and not re.search(NUMERIC_DURATION, phrase) and not re.search(r"看(?:优惠|促销|满减|购物|活动)?(?:规则|攻略)", clause):
            return False, "excluded:generic_looked_for_a_long_time_template", evidence
        if rule in ("study_duration", "measured_study_duration") and phrase.startswith(("看", "观察", "盯")) and re.search(NUMERIC_DURATION, phrase):
            if re.match(r"(?:搞笑|娱乐|综艺|游戏|搞怪|跳舞|唱歌|短|小){0,3}(?:视频|电影|电视剧|动画|综艺|直播)", post):
                return False, "excluded:content_viewing_object_not_shopping_monitoring", evidence
            # Numeric duration is explicit labor; still require a commerce
            # monitoring object, not merely a promotion word near a video.
            if not re.search(PRODUCT + r"|热水器|冰箱|电视|空调|价格|价钱|购物车|京东|淘宝|天猫|拼多多|优惠券|满减|凑单", clause):
                return False, "excluded:measured_looking_without_shopping_monitoring_object", evidence
        return True, "included:explicit_shopping_time_labor_with_local_anchor", evidence
    if component == "c07":
        evidence["anchors"] = _anchors(text, start, end, SHOP)
        if not evidence["anchors"]:
            return False, "excluded:no_shopping_time_context_within_45", evidence
        if re.search(r"(?:不是|并非|谁说)(?:我|我们)?$", pre):
            return False, "excluded:denied_time_scarcity", evidence
        if rule == "opportunity_cost" and _denied_time(pre):
            return False, "excluded:denied_time_opportunity_cost", evidence
        # Co-occurring shopping somewhere in the comment cannot license unrelated life scarcity.
        if re.search(r"(?:没(?:有)?时间|没(?:有)?空|无暇|抽不出时间).{0,4}(?:陪孩子|照顾孩子|睡觉|吃饭|工作|上班|运动|看病)", clause):
            return False, "excluded:generic_life_scarcity_not_shopping", evidence
        return True, "included:shopping_time_scarcity_or_opportunity_cost", evidence
    if component == "c12":
        if _rejected_report(pre, post):
            return False, "excluded:explicitly_rejected_price_rise_report", evidence
        if re.search(r"(?:工资|薪水|身高|体重|血压|气温|温度|成绩|积分).{0,6}$", pre):
            return False, "excluded:nonprice_quantity_increase", evidence
        if _pre_negated(pre) or re.search(r"(?:是否|会不会|有没有)$", pre):
            return False, "excluded:negated_or_unasserted_price_rise", evidence
        if re.search(r"^(?:了)?(?:吗|么)|(?:会不会|是否|有没有).{0,8}(?:涨|提)价", post + " " + clause):
            return False, "excluded:price_rise_question_without_assertion", evidence
        if re.search(r"(?:如果|假如|要是|万一|希望|担心|怕).{0,12}$", pre) or re.match(r"(?:的话|就)", post):
            return False, "excluded:conditional_or_hypothetical_price_rise", evidence
        return True, "included:price_rise_or_raise_then_discount_allegation_not_verified_price", evidence
    if component == "c16":
        if _rejected_report(pre, post):
            return False, "excluded:explicitly_rejected_poor_value_report", evidence
        if rule == "ironic_effort_reward":
            if re.search(r"(?:挺好|真香|很划算|很值|值了|很满意|太棒|赚大了)", text[end:min(len(text),end+16)]):
                return False, "excluded:ironic_template_overridden_by_explicit_positive_appraisal", evidence
            return True, "included:conventional_ironic_disproportionate_effort_reward_construction", evidence
        if re.search(r"(?:不会|绝不|不让|不再|避免|防止|拒绝|不用|无需|不必|别|不要|没有|没|不|未)(?:再|会|让你|让人|使你|使人|多|白白){0,4}$", pre) or (phrase.startswith(("不", "没", "省不")) and re.search(r"(?:不是|并非|没有说)$", pre)):
            return False, "excluded:negated_extra_spend_or_denied_bad_value", evidence
        if re.search(r"(?:不会|保证不|绝不|避免|防止|不让).{0,12}(?:多花|冤枉钱)", clause) and rule == "extra_spending":
            return False, "excluded:promotional_guarantee_or_avoidance_advice", evidence
        if re.search(r"(?:是否|会不会|是不是)$", pre) or re.match(r"(?:了)?(?:吗|么)", post):
            return False, "excluded:unasserted_value_question", evidence
        if re.search(r"(?:如果|假如|要是|万一|宁愿|愿意|可以|建议|应该).{0,8}$", pre) or re.match(r"(?:的话|就)", post):
            return False, "excluded:hypothetical_or_advised_extra_spending", evidence
        return True, "included:explicit_saving_failure_bad_value_or_extra_spending", evidence
    if component == "c25":
        actor, token = _actor(text, start, end)
        evidence.update(actor_flag=actor, actor_evidence=token)
        if _denied_self_purchase_report(pre):
            return False, "excluded:denied_self_purchase_report", evidence
        question_tail = re.sub(r"\[[^\]\n]{0,40}\]", "", post).rstrip()
        if re.search(r"(?:吗|么)[啊呀呢嘛吧啦]*$", question_tail) and not re.search(r"买|购买|下单|入手|拍", question_tail):
            return False, "excluded:purchase_question_after_object", evidence
        # Negation may govern an event through a locative/time/adverb phrase.
        # A completed clause before a contrast cannot negate the later purchase.
        scoped_negative = re.search(r"(?:没|没有|未|不曾|从没|从未)(?:(?:再|去|曾经|已经|实际|真的|特意|专门|怎么|一直|还|都|就)|(?:在|从|到|通过|于)[^了，,。！？?!；;]{1,10}){0,6}$", pre)
        if re.search(r"(?:还是|照样|仍然|结果|于是|终于)(?:就|直接|全|都)?$", pre):
            scoped_negative = None
        if scoped_negative or re.search(r"(?:不|别|不要)(?:再|去|曾经|已经|实际|真的)?$", pre):
            return False, "excluded:negated_purchase", evidence
        scoped_intent = re.search(r"(?:想要|想|打算|准备|计划|决定|考虑|希望|将要|要|会)(?!了)[^了]{0,18}$", pre)
        completed_transition = phrase.startswith(("已", "刚")) or re.search(r"(?:已经|刚刚|刚|终于|结果|于是)(?:就|直接|全|都)?$", pre)
        if re.search(r"(?:没|没有)(?:想到|料到)[^了]{0,10}$|(?:想|考虑|琢磨)了(?:想|一下|一会儿)?$", pre):
            scoped_intent = None
        if scoped_intent and not completed_transition:
            return False, "excluded:intended_not_completed_purchase", evidence
        elliptical_condition = re.search(r"(?:有|没|没有|给|不给)[^，,。！？?!；;]{0,20}(?:就|才)(?:我|我们|去|再|直接|另|别|的|平台|换|一|个|到|上|边){0,12}$", pre)
        realized_context = re.search(r"(?:昨天|刚才|当时|那天|已经|刚刚|于是|所以|领到|拿到|收到|有了|给了)", pre)
        if elliptical_condition and not realized_context:
            return False, "excluded:elliptical_condition_not_completed_purchase", evidence
        if re.search(r"(?:如果|假如|要是|只要|除非|假设|等我|等你|等到|等).{0,12}$", pre) or re.match(r".{0,10}(?:的话|就能|就会|就可以|就知道|就行|再说)", post):
            return False, "excluded:conditional_purchase", evidence
        if re.search(r"(?:是否|有没有|是不是|好像|似乎|可能).{0,4}$", pre) or re.match(r"(?:吗|么|没|没有|吧)(?:$|[？?])", post):
            return False, "excluded:purchase_question_or_uncertain_report", evidence
        if phrase == "拍了" and not _anchors(text, start, end, SHOP):
            return False, "excluded:photography_not_anchored_purchase", evidence
        return True, "included:completed_purchase_narration_without_transaction_verification", evidence
    if component == "c28":
        if re.search(r"不买.{0,8}(?:也不省|并不省|不省钱|不能省|省不了|没省)", clause):
            return False, "excluded:denial_that_abstention_saves", evidence
        return True, "included:abstention_saving_discourse_not_purchase_abandonment_fact", evidence
    if component == "c34":
        if phrase.startswith(("怎么", "如何")) and re.search(r"(?:你|你们|他|他们)连$", pre) and re.match(r"[^，,。！？?!；;]{0,12}都不会", post):
            return False, "excluded:other_directed_method_reproach_not_help", evidence
        if phrase.startswith("不会") and re.match(r"[^，,。！？?!；;]{0,12}(?:的人|者)(?:先|要|应该|应当|可以|就|得|别|不要|去|学)", post):
            return False, "excluded:descriptive_person_modifier_or_instruction_not_help", evidence
        if phrase.startswith("怎么") and re.search(r"(?:没|没有|不)(?:太|大|再)?$", pre):
            return False, "excluded:low_frequency_negation_not_method_question", evidence
        if phrase.startswith("不会"):
            if re.search(r"(?:都|也|还|就|才|从来|绝对|肯定|根本)?$", pre) and re.search(r"(?:不想|不要|不愿|不打算).{0,16}$", pre):
                return False, "excluded:unwillingness_not_inability_help", evidence
            if re.search(r"(?:那是|这是|就是)(?:你|你们|他|他们).{0,8}$", pre) or re.match(r"(?:就说|的?人|的?你|的?你们|就别|就不要)", post):
                return False, "excluded:rhetorical_advice_not_help_request", evidence
        evidence["anchors"] = _anchors(text, start, end, SHOP + "|" + PRODUCT)
        if not evidence["anchors"]:
            return False, "excluded:no_explicit_shopping_method_choice_or_explanation_context", evidence
        if re.search(r"(?:不用|无需|别|不要|不必)(?:再)?(?:告诉我|教我|教教我)?.{0,12}$", pre):
            return False, "excluded:negated_help_request", evidence
        if phrase in ("告诉我", "教教我") and re.search(r"(?:已经|曾经|昨天|刚才|朋友|他|她|客服|博主|作者).{0,6}$", pre) and not re.search(r"(?:请|能|可以|谁|麻烦)", pre):
            return False, "excluded:reported_help_not_request", evidence
        if re.search(r"(?<!不)(?<!没)(?:知道|懂得|明白|学会|学到)(?:了)?.{0,16}(?:怎么|如何)", clause) and not re.search(r"(?:吗|么)$|是否|请问", clause):
            return False, "excluded:declarative_knowledge_not_request", evidence
        if re.search(r"(?:教你|教大家|告诉你|告诉大家|教程来了|教程如下)", clause) and not re.search(r"请问|求助|求解", clause):
            return False, "excluded:instruction_or_tutorial_not_request", evidence
        if re.match(r"(?:都|也)(?:不|可以|一样|行|好)", post):
            return False, "excluded:concessive_method_statement_not_request", evidence
        if re.search(r"(?:为什么|为何).{0,12}(?:复杂|麻烦|套路)", clause) and not re.search(r"请问|求解|教教|求助", clause):
            return False, "excluded:rhetorical_complaint_not_clear_help_request", evidence
        return True, "included:explicit_shopping_method_choice_or_explanation_help_request", evidence
    if component == "c41":
        if rule == "product_effect":
            if re.search(r"(?:优惠券|券|红包|积分|活动规则|满减规则|保价|价保)(?:的|现在|已经|真的|根本|完全|好像|确实|也|都|还|就|没什么|不太|没有什么|没|没有|不){0,4}$", pre):
                return False, "excluded:promotion_or_aftersales_object_not_product_effect", evidence
            if phrase == "没用" and re.match(r"(?:完|过|光|尽|掉|上|到)", post):
                return False, "excluded:usage_or_inventory_not_product_effect", evidence
            evidence["anchors"] = _anchors(text, start, end, PRODUCT)
            if not evidence["anchors"]:
                return False, "excluded:effect_or_usability_without_product_anchor", evidence
            if re.search(r"登录|登[录陆]|手机号|电话.{0,6}登录|(?:促销|活动|宣传|视频|营销|教学)效果", clause):
                return False, "excluded:account_rule_or_content_effect_not_product_quality", evidence
        if rule == "quality_authenticity":
            if phrase == "质量" and (re.search(r"(?:空气|睡眠|生活|教学|服务|售后|物流|客服|视频|画面|人口)(?:的)?$", pre) or re.match(r"(?:的)?(?:生活(?!用品|用具)|睡眠|服务|售后|教学|人口|客服|物流)", post)):
                return False, "excluded:nonproduct_quality_or_promotion_expiry", evidence
            if phrase == "过期" and re.search(r"(?:券|活动|红包|积分)[^，,。！？?!；;]{0,5}$", pre):
                return False, "excluded:nonproduct_quality_or_promotion_expiry", evidence
        return True, "included:product_quality_effect_authenticity_topic_any_polarity", evidence
    if component == "c42":
        if rule == "customer_service":
            evidence["anchors"] = _anchors(text, start, end, SHOP + r"|" + RULES["c42"][0][1])
            if not evidence["anchors"]:
                return False, "excluded:customer_service_without_commerce_or_aftersales_anchor", evidence
        return True, "included:delivery_aftersales_price_protection_topic_any_polarity", evidence
    if component == "c43":
        if phrase.startswith(("跟着", "照着")) and _follow_buy_regret(text, start, end):
            return False, "excluded:following_purchase_with_immediate_regret_not_endorsement", evidence
        if _denied_self_purchase_report(pre):
            return False, "excluded:denied_following_purchase_report", evidence
        # Negation and requests govern the proposition, not just the immediately
        # preceding character. Bound that scope by punctuation and explicit
        # contrast markers so another proposition does not veto an endorsement.
        contrasts = list(re.finditer(r"但是|不过|然而|可是|但", clause))
        scope_left, scope_right = left, right
        for boundary in contrasts:
            a, b = left + boundary.start(), left + boundary.end()
            if b <= start:
                scope_left = b
            elif a >= end:
                scope_right = a
                break
        clause = text[scope_left:scope_right]
        pre = text[max(scope_left, start-24):start]
        post = text[end:min(scope_right, end+24)]
        evidence["endorsement_scope_span"] = [scope_left, scope_right]
        # Bracketed emoji do not turn a question particle into an assertion.
        question_post = re.sub(r"\[[^\]\n]{0,40}\]", "", post).rstrip()
        if re.match(r"(?:吗|么)", question_post):
            return False, "excluded:question_particle_before_emoji_not_endorsement", evidence
        double_negative = bool(re.search(r"(?:并不是|不是|并非|不可能|没有理由|没理由)不$", pre))
        obligatory_endorsement = phrase == "推荐" and bool(re.search(r"(?:不得不|不能不)$", pre))
        evidence["explicit_double_negative_affirmation"] = double_negative or obligatory_endorsement
        if re.search(r"(?:不认为|不觉得|没觉得|没有觉得|不相信|不敢说|不能说|不能称|不能算|谈不上|称不上|算不上|说不上|难说|很难说|不能确定|无法确定|不能保证|不保证|不确定|没法保证|不见得|未必|不一定|不建议|没有人认为|没人认为|没有人觉得|没人觉得|并非所有|不是所有|并不是所有|并非每|不是每|没有理由说|没理由说).{0,18}$", pre):
            return False, "excluded:scoped_negative_attitude_uncertainty_or_advice", evidence
        if re.search(r"(?:(?:能|能不能|可以|可不可以|可否|能否)(?:给|帮)(?:我|我们|大家)|(?:你|您)(?:能|可以|可否|能否)(?:给|帮)?(?:我|我们)?|给我|帮我|帮我们|麻烦你|麻烦您)(?:再|先|来|一下|个|一个|帮忙){0,4}$", pre):
            return False, "excluded:directed_modal_request_not_endorsement", evidence
        if re.search(r"(?:能|可以|可否|能否).{0,24}(?:不|否)$", clause):
            return False, "excluded:colloquial_modal_question_not_endorsement", evidence
        if re.search(r"(?:不能|不可|不应|不应该|不该|不宜|不适合)(?:再|直接|轻易|随便|就|这样|这么|那么){0,4}$", pre):
            return False, "excluded:modal_prohibition_not_endorsement", evidence
        if re.search(r"(?:叫|名叫|店名是|品牌名是|网名是|昵称是)$", pre):
            return False, "excluded:explicit_naming_context_not_endorsement", evidence
        if re.search(r"什么值得买|值得买app|值得买网站|值得买平台|值得买软件|信任度|可信度|信任危机|推荐算法|推荐机制|推荐系统|推荐页面|靠谱哥|靠谱姐|靠谱君|靠谱先生|靠谱旗舰店|靠谱专卖店", clause) and (phrase in ("值得买", "推荐", "信任", "可信", "靠谱")):
            return False, "excluded:proper_name_or_metalinguistic_term", evidence
        if (_pre_negated(pre) and not (double_negative or obligatory_endorsement)) or (phrase == "推荐" and re.search(r"(?:不|并非)(?:太|很)?(?:值得|建议)$", pre)) or re.search(r"(?:靠不靠谱|可不可信|推不推荐|信不信任)", clause):
            return False, "excluded:negated_or_alternative_question_endorsement", evidence
        if re.search(r"(?:求|请|能否|能不能|有没有|是否|是不是|哪家|哪个|谁|听说|据说|别人说|人家说).{0,6}$", pre) or re.search(r"(?:吗|么)[啊呀呢嘛吧啦]*$|\?", clause) or text[scope_right:scope_right+1] in ("?", "？"):
            return False, "excluded:request_question_or_hearsay_not_affirmative_endorsement", evidence
        if re.search(r"(?:想|希望|想要|打算|准备|要找|寻找|但愿|不确定|不知道|好像|似乎).{0,8}$", pre) or re.search(r"(?:如果|假如|要是|只要|万一).{0,16}$", pre) or re.match(r"(?:点|一些|的话|就(?:买|下单|入手|会)|才(?:买|下单|入手|会))", post):
            return False, "excluded:desired_uncertain_or_conditional_endorsement", evidence
        if phrase == "信任" and not double_negative and not (re.search(r"(?:我|我们|真心|非常|很|完全|绝对|一直|值得)$", pre) or re.match(r"(?:这家|这个|你|您|他|她|博主)", post)):
            return False, "excluded:trust_keyword_without_affirmative_form", evidence
        if phrase == "推荐" and not (double_negative or obligatory_endorsement) and not (re.search(r"(?:我|本人|强烈|墙裂|值得|真心|非常|必须|特别|确实|很|十分)$", pre) or re.match(r"(?:给大家|大家|你们|给你|给您|这家|这个|这款)", post)):
            return False, "excluded:recommendation_keyword_without_explicit_affirmative_form", evidence
        if "抄" in phrase:
            # Bare copying imperatives and recruiting for guides are instruction,
            # not affirmative recommendation. Habitual first-person reliance or
            # completed copying can express endorsement; retain that distinction.
            if re.search(r"数学|语文|英语|考试|课堂|同学作业", clause) and not _anchors(text, start, end, SHOP):
                return False, "excluded:academic_copying_not_purchase_endorsement", evidence
            habitual_self = bool(re.search(r"(?:我|我们|本人)(?:每次|每年|一直|经常|总是|都是|都|就|直接|放心|只){1,5}$", pre) or re.search(r"(?:每次|每年|一直|经常)(?:我|我们)(?:都|是|直接|放心){0,4}$", pre))
            completed_copy = bool(re.match(r"(?:了|过|完)", post))
            explicit_appraisal = bool(re.search(r"(?:放心|值得|推荐大家|推荐你们)$", pre))
            if not (habitual_self or completed_copy or explicit_appraisal):
                return False, "excluded:copying_instruction_or_recruitment_without_endorsement", evidence
            if not (_anchors(text, start, end, SHOP) or habitual_self or explicit_appraisal):
                return False, "excluded:copy_homework_without_purchase_endorsement_context", evidence
        return True, "included:affirmative_trust_or_recommendation_endorsement", evidence
    raise AssertionError(component)


def label_text(text: str, record_id=None):
    normalized, mapping = normalize(text)
    hits = []
    for component in COMPONENTS:
        seen = set()
        for rule, pattern in RULES[component]:
            for match in re.finditer(pattern, normalized):
                start, end = match.span()
                if (start, end) in seen:
                    continue
                seen.add((start, end))
                qualifies, reason, evidence = _qualify(component, rule, normalized, start, end)
                orig_start, orig_end = mapping[start], mapping[end-1] + 1
                hits.append({"component": component, "rule_id": f"{component}.{rule}",
                    "normalized_span": [start, end], "normalized_phrase": match.group(),
                    "original_span": [orig_start, orig_end], "original_phrase": text[orig_start:orig_end],
                    "qualifies": qualifies, "reason": reason, **evidence})
    # Count independent phrase occurrences, not overlapping regex alternatives.
    # Within each component prefer longest then earliest span; excluded candidates
    # never suppress qualifying ones. Retain suppressed records for the audit.
    for component in COMPONENTS:
        selected = []
        for hit in sorted([h for h in hits if h["component"] == component and h["qualifies"]], key=lambda h: (-(h["normalized_span"][1]-h["normalized_span"][0]), h["normalized_span"][0], h["rule_id"])):
            a,b = hit["normalized_span"]
            if any(a < d and c < b for c,d in selected):
                hit["qualifies"] = False
                hit["reason"] = "excluded:overlapping_qualifying_phrase_deduplicated"
            else:
                selected.append((a,b))
    hits.sort(key=lambda h: (h["component"], *h["normalized_span"], h["rule_id"]))
    counts = {c: sum(h["qualifies"] for h in hits if h["component"] == c) for c in COMPONENTS}
    labels = {c: int(counts[c] > 0) for c in COMPONENTS}
    outcomes = {y: int(any(labels[c] for c in cs)) for y,cs in OUTCOMES.items()}
    return {"implementation_version": VERSION, "record_id": record_id,
        "normalized_text": normalized, "component_labels": labels,
        "component_hit_counts": counts,
        "component_decision_reasons": {c: ("at_least_one_qualifying_phrase" if labels[c] else "all_lexical_candidates_excluded" if any(h["component"] == c for h in hits) else "no_lexical_candidate") for c in COMPONENTS},
        "outcome_labels": outcomes,
        "outcome_hit_counts": {y: sum(counts[c] for c in cs) for y,cs in OUTCOMES.items()},
        "purchase_actor_flags": sorted(set(h["actor_flag"] for h in hits if h["component"] == "c25" and h["qualifies"])),
        "hits": hits}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Authorized development JSONL only; must contain text")
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    with args.input.open(encoding="utf-8") as f, args.output.open("w",encoding="utf-8") as out:
        for line in f:
            if line.strip():
                row=json.loads(line)
                result=label_text(row["text"], row.get("review_id",row.get("record_id")))
                out.write(json.dumps(result,ensure_ascii=False,sort_keys=True)+"\n")

if __name__ == "__main__":
    main()
