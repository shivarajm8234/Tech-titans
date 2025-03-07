from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from app.models.user import User
from app.models.post import Post
from app import db
from datetime import datetime, timedelta
from sqlalchemy import func

admin = Blueprint('admin', __name__, url_prefix='/admin')

# Admin access decorator
def admin_required(f):
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You do not have permission to access this page.', 'danger')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return login_required(decorated_function)

@admin.route('/')
@admin_required
def dashboard():
    """Admin dashboard with statistics"""
    # Get user statistics
    total_users = User.query.count()
    total_admins = User.query.filter_by(is_admin=True).count()
    total_regular_users = total_users - total_admins
    
    # Get new users in the last 7 days
    week_ago = datetime.utcnow() - timedelta(days=7)
    new_users_last_week = User.query.filter(User.created_at >= week_ago).count()
    
    # Get active users (logged in within the last 7 days)
    active_users = User.query.filter(User.last_login >= week_ago).count()
    
    # Get post statistics
    total_posts = Post.query.count()
    authentic_posts = Post.query.filter_by(is_authentic=True).count()
    
    # Get posts in the last 7 days
    new_posts_last_week = Post.query.filter(Post.created_at >= week_ago).count()
    
    # Get top users by post count
    top_users = db.session.query(
        User.username, 
        func.count(Post.id).label('post_count')
    ).join(Post).group_by(User.id).order_by(func.count(Post.id).desc()).limit(5).all()
    
    stats = {
        'total_users': total_users,
        'total_admins': total_admins,
        'total_regular_users': total_regular_users,
        'new_users_last_week': new_users_last_week,
        'active_users': active_users,
        'total_posts': total_posts,
        'authentic_posts': authentic_posts,
        'new_posts_last_week': new_posts_last_week,
        'top_users': top_users
    }
    
    return render_template('admin/dashboard.html', stats=stats)

@admin.route('/users')
@admin_required
def user_management():
    """User management page"""
    users = User.query.all()
    return render_template('admin/users.html', users=users)

@admin.route('/users/<int:user_id>/toggle_admin', methods=['POST'])
@admin_required
def toggle_admin(user_id):
    """Toggle admin status for a user"""
    user = User.query.get_or_404(user_id)
    
    # Prevent removing admin status from yourself
    if user.id == current_user.id:
        flash('You cannot remove your own admin status.', 'danger')
        return redirect(url_for('admin.user_management'))
    
    user.is_admin = not user.is_admin
    db.session.commit()
    
    flash(f'Admin status for {user.username} has been {"granted" if user.is_admin else "revoked"}.', 'success')
    return redirect(url_for('admin.user_management'))

@admin.route('/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    """Delete a user"""
    user = User.query.get_or_404(user_id)
    
    # Prevent deleting yourself
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.user_management'))
    
    username = user.username
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User {username} has been deleted.', 'success')
    return redirect(url_for('admin.user_management'))

@admin.route('/posts')
@admin_required
def post_management():
    """Post management page"""
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template('admin/posts.html', posts=posts)

@admin.route('/posts/<int:post_id>/delete', methods=['POST'])
@admin_required
def delete_post(post_id):
    """Delete a post"""
    post = Post.query.get_or_404(post_id)
    db.session.delete(post)
    db.session.commit()
    
    flash('Post has been deleted.', 'success')
    return redirect(url_for('admin.post_management'))
