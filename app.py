import os
import re
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# --------------------------
# Helpers
# --------------------------

CHANNEL_URL_PATTERNS = [
    r'(?:https?://)?(?:www\.)?youtube\.com/channel/(?P<id>UC[0-9A-Za-z_-]{21}[AQgw])',
    r'(?:https?://)?(?:www\.)?youtube\.com/@(?P<handle>[A-Za-z0-9_.-]+)',
    r'(?:https?://)?(?:www\.)?youtube\.com/c/(?P<custom>[A-Za-z0-9_.-]+)'
]

def extract_channel_identifier(text: str) -> Dict[str, str]:
    text = text.strip()
    for pat in CHANNEL_URL_PATTERNS:
        m = re.match(pat, text)
        if m:
            d = m.groupdict()
            for k, v in d.items():
                if v:
                    return {k: v}
    if re.match(r'^UC[0-9A-Za-z_-]{22}$', text):
        return {"id": text}
    if re.match(r'^[A-Za-z0-9_.-@]+$', text):
        return {"handle": text.lstrip('@')}
    return {}

@st.cache_data(show_spinner=False)
def yt_build(api_key: str):
    return build('youtube', 'v3', developerKey=api_key)

def resolve_channel_id(youtube, identifier: Dict[str, str]) -> Optional[Dict[str, Any]]:
    try:
        if 'id' in identifier:
            resp = youtube.channels().list(part='snippet,statistics,contentDetails', id=identifier['id']).execute()
        elif 'handle' in identifier:
            resp = youtube.channels().list(part='snippet,statistics,contentDetails', forHandle=identifier['handle']).execute()
        elif 'custom' in identifier:
            s = youtube.search().list(part='snippet', q=identifier['custom'], type='channel', maxResults=1).execute()
            items = s.get('items', [])
            if not items:
                return None
            cid = items[0]['snippet']['channelId']
            resp = youtube.channels().list(part='snippet,statistics,contentDetails', id=cid).execute()
        else:
            return None
        items = resp.get('items', [])
        if not items:
            return None
        return items[0]
    except HttpError as e:
        st.error(f'YouTube API error while resolving channel: {e}')
        return None

@st.cache_data(show_spinner=False)
def fetch_recent_videos(youtube, channel_id: str, max_results: int = 200) -> pd.DataFrame:
    video_ids: List[str] = []
    next_page = None
    remaining = max_results
    while remaining > 0:
        page_size = min(50, remaining)
        resp = youtube.search().list(
            part='id',
            channelId=channel_id,
            maxResults=page_size,
            order='date',
            type='video',
            pageToken=next_page
        ).execute()
        for it in resp.get('items', []):
            vid = it['id'].get('videoId')
            if vid:
                video_ids.append(vid)
        next_page = resp.get('nextPageToken')
        if not next_page:
            break
        remaining -= page_size

    if not video_ids:
        return pd.DataFrame()

    rows = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        resp = youtube.videos().list(part='snippet,statistics,contentDetails', id=','.join(chunk)).execute()
        for v in resp.get('items', []):
            sn = v['snippet']
            stc = v.get('statistics', {})
            cd = v.get('contentDetails', {})
            thumbs = sn.get('thumbnails', {})
            thumb_url = thumbs.get('medium', thumbs.get('default', {})).get('url', '')
            rows.append({
                'video_id': v['id'],
                'title': sn.get('title'),
                'publishedAt': sn.get('publishedAt'),
                'views': int(stc.get('viewCount', 0)),
                'likes': int(stc.get('likeCount', 0)),
                'comments': int(stc.get('commentCount', 0)),
                'duration': cd.get('duration', ''),
                'thumbnail': thumb_url,
                'video_url': f"https://www.youtube.com/watch?v={v['id']}",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df = df.sort_values('publishedAt', ascending=False).reset_index(drop=True)
    return df

def human_int(n: float) -> str:
    n = float(n)
    for unit in ['', 'K', 'M', 'B']:
        if abs(n) < 1000:
            return f'{n:.0f}{unit}'
        n /= 1000.0
    return f'{n:.0f}B'

def kpis_from_df(df: pd.DataFrame, subs: int) -> Dict[str, Any]:
    if df.empty:
        return {}
    df_nonzero = df[df['views'] > 0]
    engagement_pct = ((df['likes'] + df['comments']) / df['views'].replace(0, pd.NA)).dropna().mean()
    ctr_proxy = (df_nonzero['views'] / max(subs, 1)).mean() if not df_nonzero.empty else 0
    return {
        'Total videos': len(df),
        'Median views': int(df['views'].median()),
        'Avg views': int(df['views'].mean()),
        'Max views': int(df['views'].max()),
        'Avg engagement %': round(float(engagement_pct) * 100, 2) if pd.notna(engagement_pct) else 0.0,
        'CTR proxy (views/subs)': round(float(ctr_proxy), 3),
    }

def simple_recommendations(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ['No videos found to analyze.']
    recs = []
    recent = df.head(min(30, len(df)))
    dayperf = recent.groupby(recent['publishedAt'].dt.day_name())['views'].mean().sort_values(ascending=False)
    if not dayperf.empty:
        recs.append(f'Best day (last {len(recent)} videos): {dayperf.index[0]}')
    top3 = recent.nlargest(3, 'views')[['title', 'views']].values.tolist()
    for i, (t, v) in enumerate(top3, 1):
        recs.append(f'Top {i} recent: "{t}" — {human_int(v)} views')
    span_days = (recent['publishedAt'].max() - recent['publishedAt'].min()).days or 1
    freq = len(recent) / span_days * 7
    recs.append(f'Posting frequency (last {len(recent)}): ~{freq:.1f} videos/week')
    return recs

# --------------------------
# UI
# --------------------------

st.set_page_config(page_title='YouTube Channel Analyzer', page_icon='📺', layout='wide')
st.title('📺 YouTube Channel Analyzer — Pro')

with st.sidebar:
    st.header('Setup')
    api_key = st.text_input('YouTube Data API Key', value=st.secrets.get('YOUTUBE_API_KEY', ''), type='password')
    channel_input = st.text_input('Channel URL / @handle / Channel ID', placeholder='e.g., https://youtube.com/@SochVerse or UCxxxx')
    max_vids = st.slider('Max recent videos to fetch', 20, 200, 120, step=20)
    st.markdown('---')
    st.caption('Optional: filter by publish date (applies after fetching)')
    start_date, end_date = st.date_input('Date range', value=(date(2024, 1, 1), date.today()))
    go = st.button('Analyze')

if not api_key:
    st.info('Enter your YouTube API key in the sidebar to begin.')
    st.stop()

yt = yt_build(api_key)

if go and channel_input:
    ident = extract_channel_identifier(channel_input)
    if not ident:
        st.error('Could not understand the channel input. Paste a channel URL, @handle, or channel ID.')
        st.stop()
    data = resolve_channel_id(yt, ident)
    if not data:
        st.error('Channel not found.')
        st.stop()

    ch_sn = data['snippet']
    ch_stats = data['statistics']
    ch_id = data['id']
    subscribers = int(ch_stats.get('subscriberCount', 0))

    header_left, header_right = st.columns([2, 3])
    with header_left:
        st.subheader(ch_sn['title'])
        st.write(ch_sn.get('description', '')[:300] + ('…' if len(ch_sn.get('description', '')) > 300 else ''))
        m1, m2, m3 = st.columns(3)
        m1.metric('Subscribers', human_int(subscribers))
        m2.metric('Total Views', human_int(int(ch_stats.get('viewCount', 0))))
        m3.metric('Videos', human_int(int(ch_stats.get('videoCount', 0))))

    df = fetch_recent_videos(yt, ch_id, max_vids)

    # Apply date filter
    if not df.empty and isinstance(start_date, date) and isinstance(end_date, date):
        m = (df['publishedAt'].dt.date >= start_date) & (df['publishedAt'].dt.date <= end_date)
        df = df[m].reset_index(drop=True)

    with header_right:
        if df.empty:
            st.warning('No videos found for the selected range.')
        else:
            st.subheader(f'Videos in range ({len(df)})')
            st.dataframe(df[['publishedAt', 'title', 'views', 'likes', 'comments']])

    if not df.empty:
        # Derived columns
        df['engagement'] = df['likes'] + df['comments']
        df['engagement_rate_%'] = (df['engagement'] / df['views'].replace(0, pd.NA) * 100).fillna(0).round(2)
        df['ctr_proxy'] = (df['views'] / max(subscribers, 1)).round(4)

        st.markdown('---')
        k = kpis_from_df(df, subscribers)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric('Total videos analyzed', k['Total videos'])
        c2.metric('Median views', human_int(k['Median views']))
        c3.metric('Average views', human_int(k['Avg views']))
        c4.metric('Max views', human_int(k['Max views']))
        c5.metric('Avg engagement %', k['Avg engagement %'])
        c6.metric('CTR proxy (avg)', k['CTR proxy (views/subs)'])

        tabs = st.tabs(['Thumbnails', 'Views over time', 'Top 10 videos', 'Engagement distribution', 'Data & Export', 'Recommendations'])

        with tabs[0]:
            st.caption('Click a thumbnail to open the video in a new tab.')
            # Show thumbnails in a responsive grid
            n_cols = 5
            rows = (len(df) + n_cols - 1) // n_cols
            for r in range(rows):
                cols = st.columns(n_cols)
                for c in range(n_cols):
                    idx = r * n_cols + c
                    if idx >= len(df):
                        break
                    row = df.iloc[idx]
                    with cols[c]:
                        if row['thumbnail']:
                            st.markdown(f"[![thumb]({row['thumbnail']})]({row['video_url']})")
                        st.caption(f"{row['title'][:60]}{'…' if len(row['title'])>60 else ''}\n{row['publishedAt'].date()} — {human_int(row['views'])} views")

        with tabs[1]:
            st.subheader('Views over time')
            fig = px.line(df.sort_values('publishedAt'), x='publishedAt', y='views', markers=True, hover_data=['title'])
            fig.update_layout(xaxis_title='Published', yaxis_title='Views')
            st.plotly_chart(fig, use_container_width=True)

        with tabs[2]:
            st.subheader('Top 10 videos by views')
            top10 = df.nlargest(10, 'views').sort_values('views')
            fig = px.bar(top10, x='views', y='title', orientation='h', hover_data=['publishedAt', 'engagement_rate_%', 'ctr_proxy'])
            fig.update_layout(xaxis_title='Views', yaxis_title='', margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            st.subheader('Engagement distribution (likes + comments) and engagement rate')
            fig1 = px.box(df, y='engagement', points='outliers')
            fig1.update_layout(yaxis_title='Engagement (likes + comments)')
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.box(df, y='engagement_rate_%', points='outliers')
            fig2.update_layout(yaxis_title='Engagement rate %')
            st.plotly_chart(fig2, use_container_width=True)

        with tabs[4]:
            st.subheader('Download your data')
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', data=csv, file_name='youtube_channel_analysis.csv', mime='text/csv')
            try:
                # Prepare Excel in-memory
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Videos')
                st.download_button('Download Excel', data=buffer.getvalue(), file_name='youtube_channel_analysis.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            except Exception as e:
                st.warning(f'Excel export unavailable: {e}')

            st.subheader('Raw data preview')
            st.dataframe(df)

        with tabs[5]:
            st.subheader('Recommendations')
            for r in simple_recommendations(df):
                st.write('• ' + r)

else:
    st.info('Enter a channel and click **Analyze** to fetch data.')
